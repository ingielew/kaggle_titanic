from .config import common_config
from .model import xgboost as proc_xgb
from .preprocessor import app as preprocessor_app
import pandas as pd


def main():
    input_data = pd.read_csv(common_config.TRAINING_DATA_PATH)
    input_data = preprocessor_app.preprocess(input_data)
    test_data = pd.read_csv(common_config.TEST_DATA_PATH)
    common_config.init_test_pass_idx_list(test_data)
    test_data = preprocessor_app.preprocess(test_data)
    proc_xgb.main(input_data, test_data)
