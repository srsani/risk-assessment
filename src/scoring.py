"""
Author: Sohrab Sani
Date: Febr, 2021
This script used for training model on the ingested data
"""
import pandas as pd
from sklearn import metrics
import logging
from joblib import load
from datetime import datetime

from setup import Settings

logging.basicConfig(
    filename='./logs/results.log',
    level=logging.INFO,
    filemode='a+',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")


def score_model(setup):
    """
    Function for model scoring

     This function should take a trained model, load test data, 
    and calculate an F1 score for the model relative to the test data
    it should write the result to the latestscore.txt file

    Args:
        setup (_type_): _description_
    """
    logging.info("loading the testdata.csv")
    df = pd.read_csv(f"{setup.TEST_DATA_PATH}/testdata.csv")

    logging.info("preparing the test data")
    y = df.exited.values
    X = df.drop(['corporation', 'exited'], axis=1)
    logging.info(f'df test shape: {df.shape}')

    logging.info("loading the trained model")
    model = load(f"{setup.OUTPUT_MODEL_PATH}/trainedmodel.joblib")

    logging.info("Predicting test data")
    y_pred = model.predict(X)
    f1 = metrics.f1_score(y, y_pred)
    logging.info(f"f1 score: {f1}")

    logging.info("Saving scores to text file")
    with open(f"{setup.OUTPUT_MODEL_PATH}/latestscore.txt", "w") as f:
        time_stamp = datetime.now()
        f.write(f"{time_stamp} - f1 score = {f1}\n")


if __name__ == '__main__':
    logging.info("model scoring")
    setup = Settings()
    score_model(setup)
