import os

import mysql.connector
from sqlalchemy import create_engine

import pandas as pd

from dotenv import load_dotenv

load_dotenv()


db_ecommerce = mysql.connector.connect(
    host='localhost',
    user='root',
    passwd=os.environ['MYSQL_PASSWORD'],
    database='ecommerce'
)


engine = create_engine(f'mysql://root:{os.environ['MYSQL_PASSWORD']}@localhost:3306/ecommerce_datawarehouse', echo=False)
db_datawarehouse = engine.connect()


def run_mysql_etl():

    merge_query = \
"""
SELECT *
FROM customers_df AS c
INNER JOIN orders AS o ON c.customer_id = o.customer_id
INNER JOIN order_reviews AS r ON o.order_id = r.order_id
INNER JOIN order_items AS i ON o.order_id = i.order_id
INNER JOIN products AS p ON i.product_id = p.product_id
INNER JOIN order_payments AS pay ON o.order_id = pay.order_id
INNER JOIN sellers AS s ON i.seller_id = s.seller_id
INNER JOIN product_category_name AS ct ON p.product_category_name = ct.product_category_name;
"""

    df = pd.read_sql(sql=str_sql, con=db_adventureworks2012)
