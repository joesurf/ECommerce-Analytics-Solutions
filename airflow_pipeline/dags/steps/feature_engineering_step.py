import logging
from typing import Optional


LOGGER = logging.getLogger(__name__)


class FeatureEngineeringStep:
    """Feature engineering: transform features for model training and inference.
    
    Args:
        inference_mode (bool): Whether the step is used in the training or inference pipeline. 
        feature_engineering_data (FeaturesEngineeringData): Paths relative to the FeatureEngineeringStep
    """

    def __init__() -> None:
        pass

    def __call__(
            self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        batch_path: Optional[str] = None,
    ) -> None:
        pass