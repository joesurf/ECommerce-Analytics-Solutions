from pathlib import Path
import logging

import pandas as pd

from steps.config import TrainerConfig
from steps.utils.sql_connector import SQLConnector
from steps.utils.data_classes import PreprocessingData


LOGGER = logging.getLogger(__name__)


class PreprocessStep:
    """
    Preprocessing based on Exploratory Data Analysis done in `churn_prediction/Customer_Churn_EDA.ipynb`

    Args:
      inference_mode (bool): Training or inference mode.
      preprocessing_data (PreprocessingData): PreprocessingStep output paths.
    """
    def __init__(self, inference_mode: bool, preprocessing_data: PreprocessingData) -> None:
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
        LOGGER.info("Preprocessing data...")
    
        sql_connector = SQLConnector()
        raw_df = sql_connector.sql_to_df(table=data_path)

        processed_df = self._preprocess(raw_df)

        if not self.inference_mode:
            LOGGER.info("Creating train/test data...")

            train_df = processed_df.sample(
                frac=TrainerConfig.train_size, random_state=TrainerConfig.random_state
            )
            test_df = processed_df.drop(train_df.index)

            sql_connector.df_to_sql(table=self.preprocessing_data.train_path, df=train_df)
            sql_connector.df_to_sql(table=self.preprocessing_data.test_path, df=test_df)

        else:
            sql_connector.df_to_sql(table=self.preprocessing_data.batch_path, df=processed_df)

    @staticmethod
    def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocessing.
        """
        LOGGER.info("Processing data...")
        return df