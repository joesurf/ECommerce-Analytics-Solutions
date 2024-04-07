from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from steps.retrieve_step import RetrieveStep
from steps.test_mlflow import run_sample_ml_model


# setup
inference_mode = False


# steps
retrieve_step = RetrieveStep()


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
    retrieve_step_task = PythonOperator(
        task_id="retrieve",
        python_callable=retrieve_step,
    )

    test_sample_ml_model_task = PythonOperator(
        task_id="test",
        python_callable=run_sample_ml_model,
    )

    retrieve_step_task >> test_sample_ml_model_task