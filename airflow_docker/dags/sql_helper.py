import logging

from sqlalchemy import create_engine

import pandas as pd

from dotenv import load_dotenv

load_dotenv()

MYSQL_PASSWORD='Wolfer#2164' # BT4301admin



def run_mysql_etl():
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

    # engine = create_engine(f"mysql://root:{MYSQL_PASSWORD}@127.0.0.1:3306/ecommerce", echo=False)
    # db_ecommerce = engine.connect()

    engine = create_engine(f"mysql://root:{MYSQL_PASSWORD}@127.0.0.1:3306/ecommerce_datawarehouse", echo=False)
    db_ecommerce_warehouse = engine.connect()

    # df = pd.read_sql(sql=merge_query, con=db_ecommerce)

    data = {
        'Name': ['John', 'Anna', 'Peter', 'Linda'],
        'Age': [28, 35, 45, 30],
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'City': ['New York', 'Paris', 'London', 'Tokyo']
    }

    df = pd.DataFrame(data)

    df.to_sql(name='person', con=db_ecommerce_warehouse, if_exists='replace')
    db_ecommerce_warehouse.commit()

    # db_ecommerce.close()
    db_ecommerce_warehouse.close()

    


# if __name__ == '__main__':
#     run_mysql_etl()