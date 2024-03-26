from datetime import datetime, timedelta

from sql_helper import run_mysql_etl

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator


default_args = {
    'owner': 'joe',
    'retries': 1,
    'retry_delay': timedelta(minutes=2)
}

with DAG(
    dag_id='tables_inner_join_etl',
    default_args=default_args,
    description='sample',
    start_date=datetime(2024, 3, 20, 2),
    schedule_interval='@daily'
) as dag:

    run_etl = PythonOperator(
        task_id='run_etl',
        python_callable=run_mysql_etl,
        dag=dag
    )

    run_this = BashOperator(
        task_id="run_starting",
        bash_command="echo Starting ETL Process...",
        dag=dag
    )

    run_this.set_downstream(run_etl)