import os

import pandas as pd
from sqlalchemy import create_engine


class SQLConnector:
    def __init__(self):
        self.engine = create_engine(f"mysql://admin:{os.environ['MYSQL_PASSWORD']}@{os.environ['HOST']}:3306/ecommerce_datawarehouse", echo=False)

    def sql_to_df(self, table):
        db_ecommerce_warehouse = self.engine.connect()

        df = pd.read_sql(sql=f"SELECT * FROM {table}", con=db_ecommerce_warehouse)

        db_ecommerce_warehouse.close()

        print(df.head())

        return df
    
    def df_to_sql(self, table, df):
        db_ecommerce_warehouse = self.engine.connect()

        df.to_sql(name=table, con=db_ecommerce_warehouse, if_exists='replace')

        db_ecommerce_warehouse.close()

        print(df.head())

        return df