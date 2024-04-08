import os
from pathlib import Path


TABLE = "main"
RAW_PATH = f"{TABLE}_raw"


class PreprocessConfig:
    train_path = f"{TABLE}_train_processed"
    test_path = f"{TABLE}_test_processed"
    batch_path = f"{TABLE}_batch_processed"


class FeatureEngineeringConfig:
    train_path = f"{TABLE}_train_featured"
    test_path = f"{TABLE}_test_featured"
    batch_path = f"{TABLE}_batch_featured"
    encoders_path = "/opt/airflow/data/"
    base_features = []
    ordinal_features = []
    target_features = []
    target = ""


class TrainerConfig:
    model_name ="gradient-boosting"
    random_state = 42
    train_size = 0.8
    shuffle = True
    params = {
        "n_estimators": 100,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }


class ConditionConfig:
    criteria = 0.05
    metric = "roc_auc"


class MlFlowConfig:
    uri = "http://mlflow-server:5000"
    experiment_name = "churn_predictor"
    artifact_path = "model-artifact"
    registered_model_name = "churn_predictor"