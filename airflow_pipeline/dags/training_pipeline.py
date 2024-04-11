from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator


from steps.test_mlflow import run_sample_ml_model

from steps.preprocess_step import PreprocessStep
from steps.feature_engineering_step import FeatureEngineeringStep
from steps.train_step import TrainStep
from steps.condition_step import ConditionStep
from steps.utils.data_classes import PreprocessingData, FeatureEngineeringData
from steps.config import (
    PreprocessConfig,
    FeatureEngineeringConfig,
    ConditionConfig,
)


# setup
inference_mode = False

preprocessing_data = PreprocessingData(
    raw_path=PreprocessConfig.raw_path,
    processed_path=PreprocessConfig.processed_path,
)
feature_engineering_data = FeatureEngineeringData(
    processed_path=FeatureEngineeringConfig.processed_path,
    featured_path=FeatureEngineeringConfig.featured_path,
)

# steps
preprocess_step = PreprocessStep(
    inference_mode=inference_mode, 
    preprocessing_data=preprocessing_data
)
feature_engineering_step = FeatureEngineeringStep(
    inference_mode=inference_mode,
    feature_engineering_data=feature_engineering_data
)
train_step_rf = TrainStep(
    model_name='RandomForest'
)
train_step_xgb = TrainStep(
    model_name='XGBoost'
)
train_step_lr = TrainStep(
    model_name='LogisticRegression'
)
condition_step = ConditionStep(
    criteria=ConditionConfig.criteria, 
    metric=ConditionConfig.metric
)

# dag
default_args = {
    "owner": "user", 
    "depends_on_past": False,
    "retries": 0,
    "catchup": False,
}

with DAG(
    "training-pipeline", 
    default_args=default_args,
    start_date=datetime(2024, 4, 1), 
    tags=["training"], 
    schedule=None,
) as dag:
    preprocessing_task = PythonOperator(
        task_id="preprocessing",
        python_callable=preprocess_step,
    )

    feature_engineering_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering_step, 
    )

    rf_training_task = PythonOperator(
        task_id="rf_training",
        python_callable=train_step_rf,
        op_kwargs={
            "featured_path": feature_engineering_data.featured_path,
        },
    )

    xgb_training_task = PythonOperator(
        task_id="xgb_training",
        python_callable=train_step_xgb,
        op_kwargs={
            "featured_path": feature_engineering_data.featured_path,
        },
    )

    lr_training_task = PythonOperator(
        task_id="lr_training",
        python_callable=train_step_lr,
        op_kwargs={
            "featured_path": feature_engineering_data.featured_path,
        },
    )

    validation_task = PythonOperator(
        task_id="validation",
        python_callable=condition_step,
        op_kwargs={
            "random_forest": rf_training_task.output,
            "xgboost": xgb_training_task.output,
            "logistic_regression": lr_training_task.output,
        }
    )

    start_task = EmptyOperator(task_id='start_task')
    end_task = EmptyOperator(task_id='end_task')

    start_task >> preprocessing_task >> feature_engineering_task >> [rf_training_task, lr_training_task, xgb_training_task] >> validation_task >> end_task