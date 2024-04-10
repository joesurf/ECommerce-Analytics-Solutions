import logging
from typing import Literal

import mlflow

from steps.config import MlFlowConfig


LOGGER = logging.getLogger(__name__)


class ConditionStep:
    def __init__(
        self, 
        criteria: float, 
        metric: Literal["roc_auc", "precision", "recall", "accuracy"]
    ) -> None:
        self.criteria = criteria
        self.metric = metric

    def __call__(self, random_forest: dict, xgboost: dict, logistic_regression: dict) -> None:
        LOGGER.info("Comparing...")

        print(random_forest)
        print(xgboost)
        print(logistic_regression)

        # TODO: maybe consider putting the results in data warehouse for visualisation



        # TODO: think about how to replace models
        
        # LOGGER.info(f"Run_id: {mlflow_run_id}")
        # mlflow.set_tracking_uri(MlFlowConfig.uri)

        # run = mlflow.get_run(run_id=mlflow_run_id)
        # metric = run.data.metrics[self.metric]

        # registered_models = mlflow.search_registered_models(
        #     filter_string=f"name = '{model_config.registered_model_name}'"
        # )

        # if not registered_models:
        #     mlflow.register_model(
        #         model_uri=f"runs:/{mlflow_run_id}/{model_config.artifact_path}",
        #         name=model_config.registered_model_name,
        #     )
        #     LOGGER.info("New model registered.")

        # latest_registered_model = registered_models[0]
        # registered_model_run = mlflow.get_run(
        #     latest_registered_model.latest_versions[0].run_id
        # )  # TODO: Can be improved
        # registered_metric = registered_model_run.data.metrics[self.metric]

        # if metric > registered_metric * (1 + self.criteria):
        #     mlflow.register_model(
        #         model_uri=f"runs:/{mlflow_run_id}/{model_config.artifact_path}",
        #         name=model_config.registered_model_name,
        #     )
        #     LOGGER.info("Model registered as a new version.")