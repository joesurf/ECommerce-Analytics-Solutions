import logging
from typing import Literal

import mlflow

from steps.config import MlFlowConfig


LOGGER = logging.getLogger(__name__)


class ConditionStep:
    """Condition to register the model.

    Args:
        criteria (float): Coefficient applied to the metric of the model registered in the model registry.
        metric (str): Metric as a reference. Can be `["precision", "recall", or "roc_auc"]`.
    """

    def __init__(
        self, 
        criteria: float, 
        metric: Literal["roc_auc", "precision", "recall"]
    ) -> None:
        self.criteria = criteria
        self.metric = metric

    def __call__(self, mlflow_run_id: str) -> None:
        """
        Compare the metric from the last run to the model in the registry.
        if `metric_run > registered_metric*(1 + self.criteria)`, then the model is registered.
        """
        pass