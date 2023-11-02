"""
This is the main file for the API. It contains the endpoints for the API and
"""

from flask import Flask, request

import json
import os
from diagnostics import (
    model_predictions,
    dataframe_summary,
    dataframe_nulls,
    execution_time,
)
from scoring import score_model
from reporting import score_model_cm


######################Set up variables for use in our script
app = Flask(__name__)
# app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
model_path = os.path.join(config["output_model_path"])


#######################Prediction Endpoint
@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    filename = request.args.get("filename")
    return str(model_predictions(filename))


#######################Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def score():
    # check the score of the deployed model
    f_1 = str(score_model(model_path))
    score_model_cm()
    return f_1  # add return value (a single F1 score number)


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def stats():
    filename = request.args.get("filename")
    # check means, medians, and modes for each column
    return str(dataframe_summary(filename))


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnose():
    filename = request.args.get("filename")
    # check timing and percent NA values
    nulls = dataframe_nulls(filename)
    timings = execution_time()

    return {
        "nulls": nulls,
        "timings": timings,
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
