"""
Author: Sohrab Sani
Date: Febr, 2021
This script used for training model on the ingested data
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
import logging
from joblib import dump

from setup import Settings

logging.basicConfig(
    filename='./logs/results.log',
    level=logging.INFO,
    filemode='a+',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")


def train_model(setup):
    """
    Function for training the model


    _extended_summary_

    Args:
        setup (_type_): _description_
    """

    logging.info("loading df from finaldata.csv")
    df = pd.read_csv(f"{setup.OUTPUT_FOLDER_PATH}/finaldata.csv")
    X = df.drop(['corporation', 'exited'], axis=1)
    y = df['exited'].values

    # use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='auto', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)

    logging.info("model training")
    model.fit(X, y)

    dump(model, f'{setup.OUTPUT_MODEL_PATH}/trainedmodel.joblib')


if __name__ == '__main__':
    logging.info("model training")
    setup = Settings()
    train_model(setup)
