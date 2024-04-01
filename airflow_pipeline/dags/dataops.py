import os
from datetime import datetime

import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
from dotenv import load_dotenv

from airflow import DAG
from airflow.operators.python_operator import PythonOperator


load_dotenv()


def construct_dataset():
    """
    Pseudo code:
    - Define the structure of the dataset
    - Create empty dataset or initialize dataset schema
    - Define data types, columns, and any metadata
    """
   # Database connection parameters
    username = 'root'
    password = 'password'
    host = 'host.docker.internal'  # Adjusted for Docker
    port = '3306'
    database = 'olist_staging'
    
    # Try to connect to the root database to create a new database
    try:
        # Connect to MySQL server
        root_engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/')
        with root_engine.connect() as root_conn:
            # Create new database if it doesn't exist
            root_conn.execute(f"CREATE DATABASE IF NOT EXISTS {database};")
        
    except ProgrammingError as pe:
        print(f"An error occurred: {pe}")

    # Connect to the newly created database
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')

    # Function to generate table names from CSV file paths
    def generate_table_name(file_path):
        return file_path.split('/')[-1].split('.')[0]
    
    # List of CSV files
    csv_files = [
        '/opt/airflow/data/olist_customers_dataset.csv',
        '/opt/airflow/data/olist_geolocation_dataset.csv',
        '/opt/airflow/data/olist_order_items_dataset.csv',
        '/opt/airflow/data/olist_order_payments_dataset.csv',
        '/opt/airflow/data/olist_order_reviews_dataset.csv',
        '/opt/airflow/data/olist_orders_dataset.csv',
        '/opt/airflow/data/olist_products_dataset.csv',
        '/opt/airflow/data/olist_sellers_dataset.csv',
        '/opt/airflow/data/product_category_name_translation.csv'
    ]
    
    # Iterate over CSV files to load them into the database
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        table_name = generate_table_name(file_path)
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    
    print("Dataset construction complete.")


def ingest_data():
    """
    Pseudo code:
    - Connect to data sources (e.g., databases, APIs)
    - Extract data from sources
    - Store extracted data in temporary storage or staging area
    """
    # Example implementation:
    # Connect to data sources
    # Extract data
    # Store extracted data in staging area
    merge_query = \
"""
SELECT *
FROM customer AS c
    INNER JOIN orders AS o ON c.customer_id = o.customer_id
    INNER JOIN order_reviews AS r ON o.order_id = r.order_id
    INNER JOIN order_items AS i ON o.order_id = i.order_id
    INNER JOIN products AS p ON i.product_id = p.product_id
    INNER JOIN order_payments AS pay ON o.order_id = pay.order_id
    INNER JOIN sellers AS s ON i.seller_id = s.seller_id
    INNER JOIN product_category_name AS ct ON p.product_category_name = ct.product_category_name;
"""

    engine = create_engine(f"mysql://admin:{os.environ['MYSQL_PASSWORD']}@{os.environ['HOST']}:3306/ecommerce", echo=False)
    db_ecommerce = engine.connect()

    engine = create_engine(f"mysql://admin:{os.environ['MYSQL_PASSWORD']}@{os.environ['HOST']}:3306/ecommerce_datawarehouse", echo=False)
    db_ecommerce_warehouse = engine.connect()

    df = pd.read_sql(sql=merge_query, con=db_ecommerce)
    df.to_sql(name='main', con=db_ecommerce_warehouse, if_exists='replace')

    db_ecommerce.close()
    db_ecommerce_warehouse.close()  

def integrate_data():
    """
    Pseudo code:
    - Retrieve data from staging area
    - Integrate data from various sources into a unified dataset
    - Handle data conflicts and inconsistencies
    """
    # Example implementation:
    # Retrieve data from staging area
    # Integrate data into unified dataset
    # Handle conflicts and inconsistencies
    pass

def cleanse_data():
    """
    Pseudo code:
    - Cleanse and preprocess the integrated dataset
    - Handle missing values, duplicates, outliers, etc.
    """
    # Example implementation:
    # Cleanse data (e.g., remove duplicates, handle missing values)
    pass

def transform_data():
    """
    Pseudo code:
    - Apply transformations to the cleansed dataset
    - Perform aggregations, calculations, or feature engineering
    """
    # Example implementation:
    # Apply transformations (e.g., calculate new columns, aggregate data)
    pass

def capture_lineage():
    """
    Pseudo code:
    - Record metadata about the data lineage
    - Track the flow of data from source to destination
    """
    # Example implementation:
    # Record metadata about data lineage (e.g., source, transformation, destination)
    pass

def exploratory_analysis():
    """
    Pseudo code:
    - Perform exploratory data analysis (EDA) on the transformed dataset
    - Generate insights, visualizations, or summary statistics
    """
    # Example implementation:
    # Perform EDA (e.g., generate summary statistics, create visualizations)
    pass

# Define default arguments for the DAG
default_args = {
    'owner': 'data_ops_team',
    'start_date': datetime(2024, 3, 30),
    'retries': 3,
}

# Define the DAG
dag = DAG(
    'data_ops_dag',
    default_args=default_args,
    description='DataOps DAG for dataset construction, ingestion, cleansing, transformation, lineage, and exploratory analysis',
    schedule_interval='@daily',
)

# Define tasks
construct_dataset_task = PythonOperator(
    task_id='construct_dataset',
    python_callable=construct_dataset,
    dag=dag,
)

ingest_data_task = PythonOperator(
    task_id='ingest_data',
    python_callable=ingest_data,
    dag=dag,
)

integrate_data_task = PythonOperator(
    task_id='integrate_data',
    python_callable=integrate_data,
    dag=dag,
)

cleanse_data_task = PythonOperator(
    task_id='cleanse_data',
    python_callable=cleanse_data,
    dag=dag,
)

transform_data_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag,
)

capture_lineage_task = PythonOperator(
    task_id='capture_lineage',
    python_callable=capture_lineage,
    dag=dag,
)

exploratory_analysis_task = PythonOperator(
    task_id='exploratory_analysis',
    python_callable=exploratory_analysis,
    dag=dag,
)

# Define task dependencies
construct_dataset_task >> ingest_data_task >> integrate_data_task >> cleanse_data_task >> transform_data_task
transform_data_task >> capture_lineage_task >> exploratory_analysis_task
