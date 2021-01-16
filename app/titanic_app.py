from .config import common_config
from .model import xgboost as proc_xgb
import pandas as pd


def main():
    input_data = pd.read_csv(common_config.TRAINING_DATA_PATH)
    test_data = pd.read_csv(common_config.TEST_DATA_PATH)
    # print(input_data.head())
    proc_xgb.main(input_data, test_data)




