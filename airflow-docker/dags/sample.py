from datetime import datetime, timedelta



from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator



default_args = {
    'owner': 'joe',
    'retries': 5,
    'retry_delay': timedelta(minutes=2)
}

dag = DAG(
    dag_id='sample_v2',
    default_args=default_args,
    description='sample',
    start_date=datetime(2021, 7, 29, 2),
    schedule_interval='@daily'
)

run_etl = PythonOperator(
    task_id='run_etl',
    python_callback=run_mysql_etl,
    dag=dag
)