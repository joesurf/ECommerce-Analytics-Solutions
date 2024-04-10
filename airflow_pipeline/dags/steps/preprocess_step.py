from pathlib import Path
import logging

import pandas as pd

from steps.utils.sql_connector import SQLConnector
from steps.utils.data_classes import PreprocessingData


LOGGER = logging.getLogger(__name__)


class PreprocessStep:
    def __init__(self, inference_mode: bool, preprocessing_data: PreprocessingData) -> None:
        self.inference_mode = inference_mode
        self.preprocessing_data = preprocessing_data

    def __call__(self) -> None:
        # if train_only: return "Skipping processing step"

        sql_connector = SQLConnector()
        raw_df = sql_connector.sql_to_df(table=self.preprocessing_data.raw_path)

        processed_df = self._preprocess(raw_df)

        sql_connector.df_to_sql(table=self.preprocessing_data.processed_path, df=processed_df)

    @staticmethod
    def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info("Processing data...")
        return df