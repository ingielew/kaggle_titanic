from .config import common_config
from . import titanic_app
import os

def main():
    # os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    common_config.initialize_data_paths()
    titanic_app.main()
