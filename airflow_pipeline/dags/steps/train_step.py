from typing import Dict, Any
import logging

import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

from steps.utils.sql_connector import SQLConnector
from steps.config import TrainerConfig, MlFlowConfig, LogisticRegressionConfig, RandomForestConfig, XGBoostConfig, SVCConfig


LOGGER = logging.getLogger(__name__)


class TrainStep:
    """
    Training step tracking experiments with MLFlow.
    GradientBoostingClassifier
    * precision
    * recall
    * roc_auc
    
    Args:
        params (Dict[str, Any]): Parameters of the model. Have to match the model arguments.
        model_name (str, optional): Additional information for experiments tracking. Defaults to TrainerConfig.model_name.
    """

    def __init__(
            self,
            model_name: str
    ) -> None:
        self.model_name = model_name

    def __call__(self, featured_path: str) -> dict:
        sql_connector = SQLConnector()
        featured_df = sql_connector.sql_to_df(table=featured_path)

        LOGGER.info("Creating train/test data...")

        # creating train/test set
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
        y = featured_df['Churned']  

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TrainerConfig.test_size, random_state=TrainerConfig.random_state) 

        # pipeline
        categorical_features = X_train.select_dtypes(include=['object']).columns
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numerical_transformer, numerical_features),
            ])

        # TODO: Select model
        if self.model_name == "RandomForest":
            model, modelConfig = self._random_forest(X_train, y_train, X_test, y_test)

        elif self.model_name == "XGBoost":
            model, modelConfig = self._xgboost(X_train, y_train, X_test, y_test)

        elif self.model_name == "LogisticRegression":
            model, modelConfig = self._logistic_regression(X_train, y_train, X_test, y_test)

        else: # Default: LogisticRegression
            model, modelConfig = self._logistic_regression(X_train, y_train, X_test, y_test)

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred)
        }

        # Set tracking server uri for logging
        mlflow.set_tracking_uri(uri=MlFlowConfig.uri)

        # Create a new MLflow Experiment
        mlflow.set_experiment(modelConfig.experiment_name)

        # Start an MLflow run
        with mlflow.start_run():

            print("Updating mlflow")

            # Log the hyperparameters
            mlflow.log_params(model.get_params())

            # Log the loss metric
            mlflow.log_metric("roc_auc", metrics['roc_auc'])
            mlflow.log_metric("accuracy", metrics['accuracy'])
            mlflow.log_metric("precision", metrics['precision'])
            mlflow.log_metric("recall", metrics['recall'])
            mlflow.log_metric("f1", metrics['f1'])
            
            # Set a tag to identify the experiment run
            mlflow.set_tag("Training Info", f"Churn Prediction - {modelConfig.model_name} Model")
            mlflow.set_tag("mlflow.runName", modelConfig.model_name)

            # Infer the model signature
            signature = infer_signature(X_train, pipeline.predict(X_train))

            # Log the model
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path=modelConfig.artifact_path,
                signature=signature,
                input_example=X_train,
                registered_model_name=modelConfig.registered_model_name
            )

            # Note down this model uri to retrieve the model in the future for scoring
            print(model_info.model_uri)
        
            return {"mlflow_run_id": mlflow.active_run().info.run_id, "metrics": metrics}


    @staticmethod
    def _random_forest(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
        LOGGER.info("Training random forest model...")

        model = RandomForestClassifier(**RandomForestConfig.params)

        return model, RandomForestConfig

    @staticmethod
    def _xgboost(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
        LOGGER.info("Training xgboost model...")

        model = xgb.XGBClassifier(**XGBoostConfig.params)

        return model, XGBoostConfig

    @staticmethod
    def _logistic_regression(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
        LOGGER.info("Training logistic regression model...")

        model = LogisticRegression(**LogisticRegressionConfig.params)

        return model, LogisticRegressionConfig