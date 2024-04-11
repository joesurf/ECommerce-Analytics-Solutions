import os
from datetime import datetime

import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
from dotenv import load_dotenv
import re
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('rslp')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import RSLPStemmer
from textblob import TextBlob
from nltk.tokenize import word_tokenize

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


def fetch_table(engine, table_name):
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, con=engine)
    return df

def ingest_data():

    username = 'root'
    password = 'password'
    host = 'host.docker.internal'  # Adjusted for Docker
    port = '3306'
    source_database = 'olist_staging'
    target_database = 'olist_datawarehouse'

    # Create database connections
    source_engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{source_database}')
    target_engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{target_database}')

    # Function to fetch a table from the database into a pandas DataFrame

    # Fetch the tables
    customers = fetch_table(source_engine, 'olist_customers_dataset')
    orders = fetch_table(source_engine, 'olist_orders_dataset')
    order_reviews = fetch_table(source_engine, 'olist_order_reviews_dataset')
    order_items = fetch_table(source_engine, 'olist_order_items_dataset')
    products = fetch_table(source_engine, 'olist_products_dataset')
    order_payments = fetch_table(source_engine, 'olist_order_payments_dataset')
    sellers = fetch_table(source_engine, 'olist_sellers_dataset')
    category_translation = fetch_table(source_engine, 'product_category_name_translation')

    # Perform the join operations using Pandas
    merged_df = (
        customers.merge(orders, on='customer_id')
        .merge(order_reviews, on='order_id')
        .merge(order_items, on='order_id')
        .merge(products, on='product_id')
        .merge(order_payments, on='order_id')
        .merge(sellers, on='seller_id')
        .merge(category_translation, on='product_category_name', how='left')
    )

    # Load the merged DataFrame into the new database
    merged_df.to_sql(name='main', con=target_engine, if_exists='replace', index=False)

    print("Data ingestion complete.")

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
    username = 'root'
    password = 'password'
    host = 'host.docker.internal'  # Adjusted for Docker
    port = '3306'
    database = 'olist_datawarehouse'
    db_url = f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}'
    engine = create_engine(db_url)

    # Fetch the cleansed dataset from the database
    with engine.connect() as conn:
        df = pd.read_sql('SELECT * FROM main', con=conn)
    df['review_comment_title'] = df['review_comment_title'].fillna("")
    df['review_comment_message'] = df['review_comment_message'].fillna("")

    df['have_review_message'] = df['review_comment_message'].apply(lambda x: 0 if x == '' else 1)


    with engine.connect() as conn:
        df.to_sql(name='main', con=conn, if_exists='replace', index=False)
    print("Data cleaning complete.")
    
def classify_cat(x):
    categories = {
        'Furniture': ['office_furniture', 'furniture_decor', 'furniture_living_room', 'kitchen_dining_laundry_garden_furniture', 'bed_bath_table', 'home_comfort', 'home_comfort_2', 'home_construction', 'garden_tools', 'furniture_bedroom', 'furniture_mattress_and_upholstery'],
        'Electronics': ['auto', 'computers_accessories', 'musical_instruments', 'consoles_games', 'watches_gifts', 'air_conditioning', 'telephony', 'electronics', 'fixed_telephony', 'tablets_printing_image', 'computers', 'small_appliances_home_oven_and_coffee', 'small_appliances', 'audio', 'signaling_and_security', 'security_and_services'],
        'Fashion': ['fashio_female_clothing', 'fashion_male_clothing', 'fashion_bags_accessories', 'fashion_shoes', 'fashion_sport', 'fashion_underwear_beach', 'fashion_childrens_clothes', 'baby', 'cool_stuff'],
        'Home & Garden': ['housewares', 'home_confort', 'home_appliances', 'home_appliances_2', 'flowers', 'costruction_tools_garden', 'garden_tools', 'construction_tools_lights', 'costruction_tools_tools', 'luggage_accessories', 'la_cuisine', 'pet_shop', 'market_place'],
        'Entertainment': ['sports_leisure', 'toys', 'cds_dvds_musicals', 'music', 'dvds_blu_ray', 'cine_photo', 'party_supplies', 'christmas_supplies', 'arts_and_craftmanship', 'art'],
        'Beauty & Health': ['health_beauty', 'perfumery', 'diapers_and_hygiene'],
        'Food & Drinks': ['food_drink', 'drinks', 'food'],
        'Books & Stationery': ['books_general_interest', 'books_technical', 'books_imported', 'stationery'],
        'Industry & Construction': ['construction_tools_construction', 'construction_tools_safety', 'industry_commerce_and_business', 'agro_industry_and_commerce']
    }
    for category, products in categories.items():
        if x in products:
            return category
    return 'Others'

def transform_data():
    # Database connection details
    username = 'root'
    password = 'password'
    host = 'host.docker.internal'  # Adjusted for Docker
    port = '3306'
    database = 'olist_datawarehouse'
    db_url = f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}'
    engine = create_engine(db_url)

    # Fetch the cleansed dataset from the database
    with engine.connect() as conn:
        df = pd.read_sql('SELECT * FROM main', con=conn)

    # Apply transformations

    # Categorize products
    df['product_category'] = df['product_category_name_english'].apply(classify_cat)

    # Calculate product volume and drop individual dimensions
    df['product_volume'] = df['product_length_cm'] * df['product_width_cm'] * df['product_height_cm']
    df.drop(['product_length_cm', 'product_width_cm', 'product_height_cm'], axis=1, inplace=True)

    # Convert date columns to datetime
    date_columns = ['order_purchase_timestamp', 'order_delivered_customer_date',
                    'order_estimated_delivery_date', 'shipping_limit_date', 'order_delivered_carrier_date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])

    # Calculate time-related features
    df['estimated_days_since_purchase'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.days
    df['arrival_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['shipping_days'] = (df['order_delivered_customer_date'] - df['order_delivered_carrier_date']).dt.days
    df.drop(df[df['shipping_days'] < 0].index, inplace=True)  # Remove records with negative shipping days


    def remove_breakline_carriage_return(df, column_name):
        """
        Remove breaklines and carriage returns from a column in a pandas DataFrame.

        Args:
        df (pandas.DataFrame): The DataFrame containing the column.
        column_name (str): The name of the column from which to remove breaklines and carriage returns.

        Returns:
        pandas.DataFrame: The DataFrame with breaklines and carriage returns removed from the specified column.
        """
        # Define a lambda function to remove breaklines and carriage returns
        remove_func = lambda x: re.sub(r'[\n\r]', '', x) if isinstance(x, str) else x
        
        # Apply the lambda function to the specified column
        df[column_name] = df[column_name].apply(remove_func)
        
        return df
    df = remove_breakline_carriage_return(df, 'review_comment_message')
    def re_hyperlinks(text):
        """
        Args:
        ----------
        text: string object with text content to be prepared [type: str]
        """
        # Applying regex
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


        return re.sub(pattern, ' link ', text)
    df['review_comment_message'] = df['review_comment_message'].apply(re_hyperlinks)

    def re_dates(text):
        """
        Args:
        ----------
        text_list: list object with text content to be prepared [type: list]
        """
        
        # Applying regex
        pattern = r'([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
        return re.sub(pattern, ' date ', text)
    
    df['review_comment_message'] = df['review_comment_message'].apply(re_dates)

    def re_money(text):
        """
        Args:
        ----------
        text_list: list object with text content to be prepared [type: list]
        """
        
        # Applying regex
        pattern = r'[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
        return re.sub(pattern, ' dinheiro ', text)
    df['review_comment_message'] = df['review_comment_message'].apply(re_money)

    def re_numbers(text):
        """
        Args:
        ----------
        text_series: list object with text content to be prepared [type: list]
        """
        
        # Applying regex
        return re.sub('[0-9]+', ' numero ', text)
    df['review_comment_message'] = df['review_comment_message'].apply(re_numbers)

    def re_negation(text):
        """
        Args:
        ----------
        text_series: list object with text content to be prepared [type: list]
        """
        
        # Applying regex
        return re.sub('([nN][ãÃaA][oO]|[ñÑ]| [nN] )', ' negação ', text)
    df['review_comment_message'] = df['review_comment_message'].apply(re_negation)

    def re_special_chars(text):
        """
        Args:
        ----------
        text_series: list object with text content to be prepared [type: list]
        """
        
        # Applying regex
        return re.sub(r'\W', ' ', text)
    
    df['review_comment_message'] = df['review_comment_message'].apply(re_special_chars)

    def re_whitespaces(text):
        """
        Args:
        ----------
        text: string object with text content to be prepared [type: str]
        """
        # Applying regex
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[ \t]+$', '', text)
        return text
    
    df['review_comment_message'] = df['review_comment_message'].apply(re_whitespaces)
    def stopwords_removal(text, cached_stopwords=stopwords.words('portuguese')):
        """
        Remove stopwords and convert text to lowercase.

        Args:
        text: string object containing the text.
        cached_stopwords: list of stopwords to be removed (default: Portuguese stopwords).

        Returns:
        Cleaned text without stopwords and converted to lowercase.
        """
        cleaned_text = ' '.join(word.lower() for word in text.split() if word.lower() not in cached_stopwords)
        return cleaned_text
    df['review_comment_message'] = df['review_comment_message'].apply(stopwords_removal)
    stemmer = RSLPStemmer()
    # Define the function to perform stemming
    def stem_text(text):
        # Tokenize the text into words
        words = word_tokenize(text)
        # Stem each word and join them back into a string
        stemmed_words = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    df['review_comment_message'] = df['review_comment_message'].apply(stem_text)
    def calculate_sentiment(text):
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity  # polarity ranges from -1 (negative) to 1 (positive)
        return sentiment_score

# Apply the calculate_sentiment function to the 'text_column' in the DataFrame
    df['sentiment_score'] = df['review_comment_message'].apply(calculate_sentiment)
    # Save transformed dataset back to the database
    with engine.connect() as conn:
        df.to_sql(name='main_transformed', con=conn, if_exists='replace', index=False)

    print("Data transformation complete.")

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
