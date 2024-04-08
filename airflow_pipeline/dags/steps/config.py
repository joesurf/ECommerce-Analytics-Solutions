import os
from pathlib import Path


RAW_PATH = "main_raw"


class PreprocessConfig:
    train_path = "main_train"
    test_path = "main_test"
    batch_path = "main_inference"


class TrainerConfig:
    model_name ="gradient-boosting"
    random_state = 42
    train_size = 0.8
    shuffle = True
    params = {
        "n_estimators": 100,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }