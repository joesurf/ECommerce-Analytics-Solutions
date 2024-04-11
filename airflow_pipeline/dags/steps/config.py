import os
from pathlib import Path


TABLE = "main_transformed"
INFERENCE_TABLE = "inference_transformed"


class PreprocessConfig:
    raw_path = f"{TABLE}"
    processed_path = f"{TABLE}_processed"


class PreprocessInferenceConfig:
    raw_path = f"{INFERENCE_TABLE}"
    processed_path = f"{INFERENCE_TABLE}_processed"


class FeatureEngineeringConfig:
    processed_path = f"{TABLE}_processed"
    featured_path = f"{TABLE}_featured"


class FeatureEngineeringInferenceConfig:
    processed_path = f"{INFERENCE_TABLE}_processed"
    featured_path = f"{INFERENCE_TABLE}_featured"


class TrainerConfig:
    featured_path: str = f"{TABLE}_featured"
    random_state: int = 42
    train_size: float = 0.8
    test_size: float = 0.2
    shuffle: bool = True


class ConditionConfig:
    criteria = 0.8
    metric = "accuracy"


class MlFlowConfig:
    uri = "http://mlflow-server:5000"


class LogisticRegressionConfig:
    # for training
    model_name: str ="logistic-regression"
    params: dict = {
        "multi_class": 'multinomial', 
        "solver": 'lbfgs'
    }

    # for mlflow
    experiment_name: str = "churn_predictor_logistic_regression"
    artifact_path: str = "model_artifact_logistic_regression"
    registered_model_name: str = "churn_predictor_logistic_regression"


class RandomForestConfig:
    # for training
    model_name: str ="random-forest"
    params: dict = {
        "random_state": 10
    }

    # for mlflow
    experiment_name: str = "churn_predictor_random_forest"
    artifact_path: str = "model_artifact_random_forest"
    registered_model_name: str = "churn_predictor_random_forest"


class XGBoostConfig:
    # for training
    model_name: str ="xgboost"
    params: dict = {}

    # for mlflow
    experiment_name: str = "churn_predictor_xgboost"
    artifact_path: str = "model_artifact_xgboost"
    registered_model_name: str = "churn_predictor_xgboost"


class SVCConfig:
    # for training
    model_name: str ="svc"
    params: dict = {
        "probability": True, 
        "random_state": 42
    }

    # for mlflow
    experiment_name: str = "churn_predictor_svc"
    artifact_path: str = "model_artifact_svc"
    registered_model_name: str = "churn_predictor_svc"