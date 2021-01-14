from .config import common_config
from . import titanic_app


def main():
    common_config.initialize_data_paths()
    titanic_app.main()
