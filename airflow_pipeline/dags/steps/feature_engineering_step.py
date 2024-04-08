import logging
from typing import Optional

from steps.utils.data_classes import FeatureEngineeringData


LOGGER = logging.getLogger(__name__)


class FeatureEngineeringStep:
    """Feature engineering: transform features for model training and inference.
    
    Args:
        inference_mode (bool): Whether the step is used in the training or inference pipeline. 
        feature_engineering_data (FeaturesEngineeringData): Paths relative to the FeatureEngineeringStep
    """

    def __init__(
        self, 
        inference_mode: bool, 
        feature_engineering_data: FeatureEngineeringData
    ) -> None:
        self.inference_mode = inference_mode
        self.feature_engineering_data = feature_engineering_data

    def __call__(
            self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        batch_path: Optional[str] = None,
    ) -> None:
        LOGGER.info("Feature Engineering...")