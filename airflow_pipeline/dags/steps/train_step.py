from typing import Dict, Any

from steps.config import TrainerConfig, MlFlowConfig


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
        pass