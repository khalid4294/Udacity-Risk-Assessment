"""
This file should contain the code to load in a trained model and score it on a test set.
"""

from sklearn import metrics
import pandas as pd
import pickle
import os
import json


with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
model_path = os.path.join(config["output_model_path"])


def score_model(model_path, data_path=None):
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file
    with open(model_path + "/trainedmodel.pkl", "rb") as f:
        model = pickle.load(f)

    if data_path is None:
        print("No data path provided, using test data path")
        df = pd.read_csv(test_data_path + "/testdata.csv")
    else:
        df = pd.read_csv(data_path)

    X_test = df.drop(["exited", "corporation"], axis=1).values.reshape(
        -1, len(df.columns) - 2
    )
    y_test = df["exited"].values.reshape(-1, 1)

    y_pred = model.predict(X_test)
    f1_score = metrics.f1_score(y_test, y_pred)

    # write f1 score to file output_model_path
    with open(model_path + "/latestscore.txt", "w") as f:
        f.write(str(f1_score))

    return f1_score


if __name__ == "__main__":
    score_model(model_path)
