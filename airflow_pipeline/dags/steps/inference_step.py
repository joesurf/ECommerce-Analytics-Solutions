import logging
from pathlib import Path
import json
from typing import List

import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler

from steps.utils.sql_connector import SQLConnector
from steps.config import MlFlowConfig, RandomForestConfig

LOGGER = logging.getLogger(__name__)


class InferenceStep:
    def __call__(self, featured_path: str) -> List[int]:
        model = self._load_model(
            registered_model_name=RandomForestConfig.registered_model_name
        )

        sql_connector = SQLConnector()
        featured_df = sql_connector.sql_to_df(table=featured_path)       
        
        X = featured_df[[
            'review_score', 'price', 'freight_value', 'payment_installments',
            'payment_value', 'customer_state_MG',
            'customer_state_RJ', 'customer_state_SP', 'payment_type_boleto',
            'payment_type_credit_card', 'payment_type_debit_card',
            'seller_state_MG',
            'seller_state_PR', 'seller_state_SP',
            'product_category_Beauty & Health',
            'product_category_Books & Stationery', 'product_category_Electronics',
            'product_category_Entertainment', 'product_category_Fashion',
            'product_category_Food & Drinks', 'product_category_Furniture',
            'product_category_Industry & Construction'
        ]]

        # standardise numerical variables
        numerical_columns = ['review_score', 'price', 'freight_value', 'payment_installments', 'payment_value']

        scaler = StandardScaler()

        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])


        if model:
            # Transform np.ndarray into list for serialization
            prediction = model.predict(X).tolist()
            LOGGER.info(f"Prediction: {prediction}")
            return json.dumps(prediction)
        else:
            LOGGER.warning(
                "No model used for prediction. Model registry probably empty."
            )

    @staticmethod
    def _load_model(registered_model_name: str):
        mlflow.set_tracking_uri(MlFlowConfig.uri)
        models = mlflow.search_registered_models(
            filter_string=f"name = '{registered_model_name}'"
        )
        LOGGER.info(f"Models in the model registry: {models}")
        if models:
            latest_model_version = models[0].latest_versions[0].version
            LOGGER.info(
                f"Latest model version in the model registry used for prediction: {latest_model_version}"
            )
            model = mlflow.sklearn.load_model(
                model_uri=f"models:/{registered_model_name}/{latest_model_version}"
            )
            return model
        else:
            LOGGER.warning(
                f"No model in the model registry under the name: {MlFlowConfig.registered_model_name}."
            )