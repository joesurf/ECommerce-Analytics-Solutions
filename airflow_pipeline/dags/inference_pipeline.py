from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator


from steps.test_mlflow import run_sample_ml_model

from steps.preprocess_step import PreprocessStep
from steps.feature_engineering_step import FeatureEngineeringStep
from steps.inference_step import InferenceStep
from steps.utils.data_classes import PreprocessingData, FeatureEngineeringData
from steps.config import (
    PreprocessInferenceConfig,
    FeatureEngineeringInferenceConfig,
)


# setup
inference_mode = True

preprocessing_data = PreprocessingData(
    raw_path=PreprocessInferenceConfig.raw_path,
    processed_path=PreprocessInferenceConfig.processed_path,
)
feature_engineering_data = FeatureEngineeringData(
    processed_path=FeatureEngineeringInferenceConfig.processed_path,
    featured_path=FeatureEngineeringInferenceConfig.featured_path,
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
inference_step = InferenceStep()


# dag
default_args = {
    "owner": "user", 
    "depends_on_past": False,
    "retries": 0,
    "catchup": False,
}

with DAG(
    "inference-pipeline", 
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

    inference_task = PythonOperator(
        task_id="inference",
        python_callable=inference_step,
        op_kwargs={
            "featured_path": feature_engineering_data.featured_path,
        },
    )

    start_task = EmptyOperator(task_id='start_task')
    end_task = EmptyOperator(task_id='end_task')

    start_task >> preprocessing_task >> feature_engineering_task >> inference_task >> end_task