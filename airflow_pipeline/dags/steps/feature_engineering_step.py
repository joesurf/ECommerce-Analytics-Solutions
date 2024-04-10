import logging
import pandas as pd

from steps.utils.sql_connector import SQLConnector
from steps.utils.data_classes import FeatureEngineeringData


LOGGER = logging.getLogger(__name__)


class FeatureEngineeringStep:
    def __init__(
        self, 
        inference_mode: bool, 
        feature_engineering_data: FeatureEngineeringData
    ) -> None:
        self.inference_mode = inference_mode
        self.feature_engineering_data = feature_engineering_data

    def __call__(self) -> None:
        # if train_only: return "Skipping feature engineering step"

        sql_connector = SQLConnector()
        processed_df = sql_connector.sql_to_df(table=self.feature_engineering_data.processed_path)

        featured_df = self._feature_engineering(processed_df)

        sql_connector.df_to_sql(table=self.feature_engineering_data.featured_path, df=featured_df)

    @staticmethod
    def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info("Feature Engineering...")

        # Readjust categories
        df['customer_state'] = df['customer_state'].apply(lambda x: x if x in ['SP', 'MG', 'RJ'] else 'Others')
        df['seller_state'] = df['seller_state'].apply(lambda x: x if x in ['SP', 'MG', 'PR'] else 'Others')

        # One hot encoding
        columns_to_encode = ['customer_state', 'payment_type', 'seller_state', 'product_category']
        df_encoded = pd.get_dummies(df, columns=columns_to_encode)

        columns_to_drop = ['customer_state_Others', 'payment_type_voucher', 'seller_state_Others', 'product_category_Home & Garden']
        df_encoded = df_encoded.drop(columns_to_drop, axis=1)


        # create churn variable
        latest_order_date = df_encoded['order_purchase_timestamp'].max()
        print("Latest order date: {}".format(latest_order_date))

        latest_purchase_df = df_encoded.groupby('customer_id')['order_purchase_timestamp'].max()
        latest_purchase_df = latest_purchase_df.reset_index().rename(columns={'order_purchase_timestamp': 'latest_purchase_date'})
        latest_purchase_df['days_since_last_purchase'] = (latest_order_date - latest_purchase_df['latest_purchase_date']).dt.days

        bins = [-1, 29, 89, 179, 359, float('inf')]  
        labels = ['0-30 days', '1-3 month', '3-6 months', '6 months-1 year', 'More than 1 year']
        latest_purchase_df['purchase_interval'] = pd.cut(latest_purchase_df['days_since_last_purchase'], bins=bins, labels=labels)

        df_encoded = pd.merge(df_encoded, latest_purchase_df, on="customer_id", how='left')

        mapping = {'More than 1 year': 1, '6 months-1 year': 0, '3-6 months': 0, '1-3 month': 0, '0-30 days': 0}
        df_encoded['Churned'] = df_encoded['purchase_interval'].map(mapping)
        df_encoded.drop(columns=['purchase_interval'], inplace=True)

        return df_encoded