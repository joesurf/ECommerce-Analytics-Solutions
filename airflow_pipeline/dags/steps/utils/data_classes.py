from dataclasses import dataclass
from typing import Optional


@dataclass
class PreprocessingData:
    raw_path: Optional[str] = None
    processed_path: Optional[str] = None


@dataclass
class FeatureEngineeringData:
    processed_path: Optional[str] = None
    featured_path: Optional[str] = None