from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import create_prediction_model
import diagnosis
import predict_exited_from_saved_model
import json
import os
from setup import Settings

from diagnostics import (model_predictions)

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

prediction_model = None


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    filepath = request.get_json()['filepath']

    df = pd.read_csv(filepath)
    df = df.drop(['corporation', 'exited'], axis=1)

    y_pred, _ = model_predictions(df)
    return jsonify(y_pred.tolist())

# Scoring Endpoint


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def stats():
    # check the score of the deployed model
    return  # add return value (a single F1 score number)

# Summary Statistics Endpoint


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    # check means, medians, and modes for each column
    return  # return a list of all calculated summary statistics

# Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def stats():
    # check timing and percent NA values
    return  # add return value for all diagnostics


if __name__ == "__main__":
    setup = Settings()
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
