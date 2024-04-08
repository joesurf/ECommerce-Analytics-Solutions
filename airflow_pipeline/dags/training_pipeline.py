from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator


from steps.test_mlflow import run_sample_ml_model

from steps.preprocess_step import PreprocessStep
from steps.feature_engineering_step import FeatureEngineeringStep
from steps.train_step import TrainStep
from steps.condition_step import ConditionStep
from steps.utils.data_classes import PreprocessingData, FeatureEngineeringData
from steps.config import (
    RAW_PATH,
    PreprocessConfig,
    FeatureEngineeringConfig,
    TrainerConfig,
    ConditionConfig,
)


# setup
inference_mode = False

preprocessing_data = PreprocessingData(
    train_path=PreprocessConfig.train_path,
    test_path=PreprocessConfig.test_path
)
feature_engineering_data = FeatureEngineeringData(
    train_path=FeatureEngineeringConfig.train_path,
    test_path=FeatureEngineeringConfig.test_path,
    encoders_path=FeatureEngineeringConfig.encoders_path,
)
target = FeatureEngineeringConfig.target

# steps
preprocess_step = PreprocessStep(
    inference_mode=inference_mode, 
    preprocessing_data=preprocessing_data
)
feature_engineering_step = FeatureEngineeringStep(
    inference_mode=inference_mode,
    feature_engineering_data=feature_engineering_data
)
train_step = TrainStep(
    params=TrainerConfig.params
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
        op_kwargs={"data_path": RAW_PATH}
    )

    feature_engineering_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering_step,
        op_kwargs={
            "train_path": preprocessing_data.train_path,
            "test_path": preprocessing_data.test_path,
        },
    )

    training_task = PythonOperator(
        task_id="training",
        python_callable=train_step,
        op_kwargs={
            "train_path": feature_engineering_data.train_path,
            "test_path": feature_engineering_data.test_path,
            "target": target
        },
    )

    validation_task = PythonOperator(
        task_id="validation",
        python_callable=condition_step,
    )

    test_sample_ml_model_task = PythonOperator(
        task_id="test",
        python_callable=run_sample_ml_model,
        op_kwargs=training_task.output,
    )

    start_task = EmptyOperator(task_id='start_task')
    end_task = EmptyOperator(task_id='end_task')


    # retrieve_step_task >> test_sample_ml_model_task
    start_task >> preprocessing_task >> feature_engineering_task >> training_task >> validation_task >> end_task