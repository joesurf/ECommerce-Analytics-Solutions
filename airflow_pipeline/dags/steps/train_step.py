from typing import Dict, Any
import logging

import mlflow

from steps.config import TrainerConfig, MlFlowConfig


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
            params: Dict[str, Any],
            model_name: str = TrainerConfig.model_name
    ) -> None:
        self.params = params
        self.model_name = model_name

    def __call__(
            self,
            train_path: str,
            test_path: str,
            target: str
        ) -> None:
        LOGGER.info("Training...")

        return {"mlflow_run_id": mlflow.active_run().info.run_id}