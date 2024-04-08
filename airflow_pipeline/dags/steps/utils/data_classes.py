from dataclasses import dataclass
from typing import Optional


@dataclass
class PreprocessingData:
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    batch_path: Optional[str] = None


@dataclass
class FeatureEngineeringData:
    encoders_path: str
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    batch_path: Optional[str] = None