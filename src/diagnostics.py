"""
Author: Sohrab Sani
Date: Febr, 2021
This script used for training model on the ingested data
"""
import pandas as pd
import numpy as np
from sklearn import metrics
import logging
import subprocess
import json
from joblib import load

from setup import Settings

from functools import wraps
from time import time
from datetime import datetime

setup = Settings()


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        _ = f(*args, **kw)
        te = time()
        duration = te-ts
        print(f'func:{f.__name__} args:[{args}, {kw}] took: {duration} sec')
        return duration
    return wrap


logging.basicConfig(
    filename='./logs/results.log',
    level=logging.INFO,
    filemode='a+',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")


def model_predictions(df_X, y):
    """
    Function to get model predictions

    read the deployed model and a test dataset, calculate predictions

    Returns:
        list: list containing all predictions
    """

    model = load(f"{setup.OUTPUT_MODEL_PATH}/trainedmodel.joblib")

    logging.info("running predictions on test_data")
    y_pred = model.predict(df_X)
    f1 = metrics.f1_score(y, y_pred)
    logging.info(f"test f1 score: {f1}")

    file = open(f"{setup.DIAGNOSTICS}/model_predictions.txt", "w")
    file.write(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    file.write(f"F1 score: {str(f1)}\n")
    for item in list(y_pred):
        file.write(f'{str(item)}\n')
    file.close()

    return y_pred, f1


def dataframe_summary():
    """
    Function to get summary statistics

    calculate summary statistics here

    Returns:
        _type_: _description_
    """
    logging.info("loading df from finaldata.csv")
    df = pd.read_csv(f"{setup.OUTPUT_FOLDER_PATH}/finaldata.csv")
    df = df.drop(['exited'], axis=1)
    df = df.select_dtypes('number')

    logging.info("calculating statistics for data")
    stat_dict = {}
    for col in df.columns:
        mean = df[col].mean()
        median = df[col].median()
        std = df[col].std()

        stat_dict[col] = {'mean': mean,
                          'median': median,
                          'std': std}

        logging.info(f"{col} mean: {mean}")
        logging.info(f"{col} median: {median}")
        logging.info(f"{col} std: {std}")

    with open(f"{setup.DIAGNOSTICS}/stat_dict.txt", "w") as f:
        f.write(
            f"\ndate: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(json.dumps(stat_dict, indent=4))
    return stat_dict


@timing
def _fun_timing(fun):
    """
    Runs ingestion.py script and measures execution time

    Returns:
        float: running time
    """

    _ = subprocess.run(['python', f'src/{fun}'], capture_output=True)
    return timing


def execution_time():
    """
    Function to get timings

    calculate timing of training.py and ingestion.py

    Returns:
        _type_: _description_
    """
    logging.info("running ingestion.py * 20")
    ingestion_time = [_fun_timing('ingestion.py') for _ in range(20)]
    [logging.info(i) for i in ingestion_time]

    training_time = []
    logging.info("running training.py * 20")
    training_time = [_fun_timing('training.py') for _ in range(20)]
    [logging.info(i) for i in training_time]

    return [{'ingest_time_mean': np.mean(ingestion_time)},
            {'train_time_mean': np.mean(training_time)}]


def outdated_packages_list():
    """
    Function to check dependencies

    check requirements.txt file using pip-outdated

    Returns:
        str: stdout of the pip-outdated command
    """
    logging.info("checking dependencies")
    dependencies = subprocess.run(['pip-outdated', 'requirements.txt'],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  encoding='utf-8')

    dd_out = dependencies.stdout.translate(
        str.maketrans('', '', ' \t\r')).split('\n')
    dd_out = [dd_out[3]] + dd_out[5:-3]
    dd_out = [s.split('|')[1:-1] for s in dd_out]

    # record the results
    file = open(f"{setup.DIAGNOSTICS}/pip_out_of_date.txt", "w")
    file.write(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    for item in dd_out:
        print(item)
        file.write(f'{str(item)}\n')
    file.close()
    return dd_out


def missing_percentage():
    """
    missing_percentage _summary_

    _extended_summary_

    Returns:
        _type_: _description_
    """
    df = pd.read_csv(f"{setup.OUTPUT_FOLDER_PATH}/finaldata.csv")
    missing_list = {col: {'percentage': perc} for col, perc in zip(
        df.columns, df.isna().sum() / df.shape[0] * 100)}
    return missing_list


if __name__ == '__main__':
    logging.info("diagnostics")
    setup = Settings()

    logging.info("loading test data")
    df = pd.read_csv(f"{setup.TEST_DATA_PATH}/testdata.csv")
    y = df['exited'].values
    X = df.drop(['corporation', 'exited'], axis=1)
    y_pred, f1 = model_predictions(df_X=X,
                                   y=y)
    stat_dict = dataframe_summary()
    execution_time()
    outdated_packages_list()
