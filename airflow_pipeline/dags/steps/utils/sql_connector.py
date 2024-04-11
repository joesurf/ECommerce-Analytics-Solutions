import os
import logging

import pandas as pd
from sqlalchemy import create_engine


LOGGER = logging.getLogger(__name__)


class SQLConnector:
    def __init__(self):
        self.engine = create_engine(f"mysql://admin:{os.environ['MYSQL_PASSWORD']}@{os.environ['HOST']}:3306/ecommerce_datawarehouse", echo=False)

    def sql_to_df(self, table):
        LOGGER.info("Converting SQL to DF...")

        db_ecommerce_warehouse = self.engine.connect()

        df = pd.read_sql(sql=f"SELECT * FROM {table}", con=db_ecommerce_warehouse)

        db_ecommerce_warehouse.close()

        return df
    
    def df_to_sql(self, table, df):
        LOGGER.info("Converting DF to SQL...")

        db_ecommerce_warehouse = self.engine.connect()

        df.to_sql(name=table, con=db_ecommerce_warehouse, if_exists='replace')

        db_ecommerce_warehouse.close()

        return df