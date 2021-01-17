import os

TRAINING_DATA_PATH = ""
TEST_DATA_PATH = ""
PREDICTIONS_PATH = ""
PASSENGER_IDX_LIST = []


def initialize_data_paths():
    global TRAINING_DATA_PATH
    TRAINING_DATA_PATH = os.path.join(os.getcwd(), 'data', 'csv_raw_input', 'train.csv')

    global TEST_DATA_PATH
    TEST_DATA_PATH = os.path.join(os.getcwd(), 'data', 'csv_raw_input', 'test.csv')

    global PREDICTIONS_PATH
    PREDICTIONS_PATH = os.path.join(os.getcwd(), 'data', 'predictions.csv')


def init_test_pass_idx_list(test_data_frame):
    global PASSENGER_IDX_LIST
    PASSENGER_IDX_LIST = test_data_frame['PassengerId']
