import requests
import logging
import json
from setup import Settings

logging.basicConfig(
    filename='./logs/results.log',
    level=logging.INFO,
    filemode='a+',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")

# Specify a URL that resolves to your workspace
URL = "http://0.0.0.0:8000"


def run_prediction():

    response_pred = requests.post(f'{URL}/prediction',
                                  json={'filepath': f"{setup.TEST_DATA_PATH}/testdata.csv"})

    return response_pred.text


def get_scoring():
    return requests.get(f'{URL}/scoring').json()


def get_summarystats():
    return requests.get(f'{URL}/summarystats').text


def get_diagnostics():
    return requests.get(f'{URL}/diagnostics').text


if __name__ == '__main__':
    logging.info("api calls")
    setup = Settings()
    logging.info("Post request /prediction for 'testdata.csv'")
    response_pred = run_prediction()

    f1_score = get_scoring()

    summarystats = get_summarystats()

    diagnostics = get_diagnostics()

    with open(f"{setup.OUTPUT_MODEL_PATH}/apireturns.txt", 'w') as file:

        file.write('predictions:\n')
        file.write(response_pred)

        file.write('\nF1 score:\n')
        file.write(str(f1_score))

        file.write('\ndf summary statistics:\n')
        file.write(summarystats)

        file.write('\ndf summary diagnostics \n')
        file.write(diagnostics)
