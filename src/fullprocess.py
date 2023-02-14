
"""
Author: Sohrab Sani
Date: February, 2021
This script used for plot and saveing confusionmatrix.png 
"""

from setup import Settings
import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion

import pandas as pd
import os
import logging
from diagnostics import model_predictions
import subprocess
import re


setup = Settings()

logging.basicConfig(
    filename='./logs/results.log',
    level=logging.INFO,
    filemode='a+',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")


def go():

    # Check and read new data
    # first, read ingestedfiles.txt
    logging.info("checking for the new data")
    df = pd.read_json(
        f"{setup.PROD_DEPLOYMENT_PATH}/ingestedfiles.txt",  encoding="utf8")
    print(df.columns)
    # determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    source_files = set(os.listdir(setup.INPUT_FOLDER_PATH))
    print(source_files)

    new_file_list = list(sorted(source_files - set(df.columns)))
    print(new_file_list)

    # if new data, proceed; otherwise,STOP
    if not new_file_list:
        logging.info("new file not found")
        return None
    if new_file_list:

        logging.info("merging data")
        subprocess.run(['python', 'src/ingestion.py'], capture_output=True)

        # Checking for model drift
        with open(f"{setup.PROD_DEPLOYMENT_PATH}/latestscore.txt") as file:
            current_f1 = float(re.findall(
                r"[-+]?(?:\d*\.*\d+)", file.read())[-1])
            print(current_f1)

        logging.info("train a new model")
        subprocess.run(['python', 'src/training.py'], capture_output=True)

        df = pd.read_csv(f"{setup.OUTPUT_FOLDER_PATH}/finaldata.csv")
        y = df['exited'].values
        X = df.drop(['corporation', 'exited'], axis=1)

        y_pred, new_f1_score = diagnostics.model_predictions(X, y)
        print(new_f1_score)

        # if no model drift STOP
        if(current_f1 >= new_f1_score):
            logging.info(" No model Drift has occurred")
            return None

        if(new_f1_score >= current_f1):
            # deploy the new model
            subprocess.run(['python', 'src/deployment.py'],
                           capture_output=True)

            # run diagnostics
            subprocess.run(['python', 'src/diagnostics.py'],
                           capture_output=True)

            # update reports
            subprocess.run(['python', 'src/reporting.py'],
                           capture_output=True)


if __name__ == '__main__':
    go()
