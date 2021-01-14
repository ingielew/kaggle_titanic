import os

TRAINING_DATA_PATH = ""
TEST_DATA_PATH = ""
PREDICTIONS_PATH = ""


def initialize_data_paths():
    global TRAINING_DATA_PATH
    TRAINING_DATA_PATH = os.path.join(os.getcwd(), 'data', 'csv_raw_input', 'test.csv')

    global TEST_DATA_PATH
    TEST_DATA_PATH = os.path.join(os.getcwd(), 'data', 'csv_raw_input', 'train.csv')

    global PREDICTIONS_PATH
    PREDICTIONS_PATH = os.path.join(os.getcwd(), 'data', 'predictions.csv')
