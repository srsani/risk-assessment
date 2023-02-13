"""
Author: Sohrab Sani
Date: February, 2023
This script used to setup settings for the current project
"""
import json
from pydantic import BaseSettings
import os

import sys
sys.path.append('..')


with open('config.json', 'r') as f:
    config = json.load(f)


class Settings(BaseSettings):

    PROJECT_PATH = os.getcwd()
    TEST_DATA_PATH = f"{PROJECT_PATH}/{config['test_data_path']}"
    INPUT_FOLDER_PATH = f"{PROJECT_PATH}/{config['input_folder_path']}"
    OUTPUT_FOLDER_PATH = f"{PROJECT_PATH}/{config['output_folder_path']}"
    OUTPUT_MODEL_PATH = f"{PROJECT_PATH}/{config['output_model_path']}"
    PROD_DEPLOYMENT_PATH = f"{PROJECT_PATH}/{config['prod_deployment_path']}"
    DIAGNOSTICS = f"{PROJECT_PATH}/{config['diagnostics']}"
