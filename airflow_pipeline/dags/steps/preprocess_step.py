from pathlib import Path

import pandas as pd

from steps.utils.data_classes import PreprocessingData

class PreprocessStep:
    """
    Preprocessing based on Exploratory Data Analysis done in `churn_prediction/Customer_Churn_EDA.ipynb`

    Args:
      inference_mode (bool): Training or inference mode.
      preprocessing_data (PreprocessingData): PreprocessingStep output paths.
    """
    def __init__(
        self,
        inference_mode: bool,
        preprocessing_data: PreprocessingData
    ) -> None:
        self.inference_mode = inference_mode
        self.preprocessing_data = preprocessing_data

    def __call__(self, data_path: Path) -> None:
        """
        Data is preprocessed then, regarding if inference=True or False:
            * False: Split data into train and test.
            * True: Data preprocessed then returned simply
        
        Args:
            data_path (Path): Input
        """
        return


    @staticmethod
    def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocessing.
        """
        return pd.DataFrame()