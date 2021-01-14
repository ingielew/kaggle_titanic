from .config import common_config
import pandas as pd


def main():
    print(common_config.TEST_DATA_PATH)
    print(common_config.TRAINING_DATA_PATH)
    input_data = pd.read_csv(common_config.TEST_DATA_PATH)
    print(input_data.head())



