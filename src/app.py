from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
# import diagnosis
# import predict_exited_from_saved_model
import json
import os
from setup import Settings
import subprocess
import re

from diagnostics import (model_predictions,
                         dataframe_summary,
                         execution_time,
                         outdated_packages_list,
                         missing_percentage,
                         )

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    filepath = request.get_json()['filepath']

    df = pd.read_csv(filepath)
    y = df.exited.values
    df_X = df.drop(['corporation', 'exited'], axis=1)
    y_pred, _ = model_predictions(df_X=df_X, y=y)
    return jsonify(y_pred.tolist())


# Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    # check the score of the deployed model
    # add return value (a single F1 score number)
    output = subprocess.run(['python', 'src/scoring.py'],
                            capture_output=True).stdout
    return output


# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summarystat():
    # check means, medians, and modes for each column
    # return a list of all calculated summary statistics
    return jsonify(dataframe_summary())


# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostic():
    # check timing and percent NA values
    # add return value for all diagnostics

    time__ = execution_time()
    req_out_dated = outdated_packages_list()
    missing__ = missing_percentage()

    req = {
        'execution_time': time__,
        'outdated_packages': req_out_dated,
        'missing_percentage': missing__
    }

    return jsonify(req)


if __name__ == "__main__":
    setup = Settings()
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)
