"""
This script is used to generate a confusion matrix for the deployed model.
"""

import pickle
import pandas as pd
from sklearn import metrics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns
import json
import os


with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
model_path = os.path.join(config["output_model_path"])
model_name = "trainedmodel.pkl"
test_dataset = "testdata.csv"


def score_model_cm():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    with open(model_path + "/" + model_name, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(test_data_path + "/" + test_dataset)
    X_test = df.drop(["exited", "corporation"], axis=1).values.reshape(
        -1, len(df.columns) - 2
    )
    y_test = df["exited"].values.reshape(-1, 1)

    y_pred = model.predict(X_test)

    # plot confusion matrix and save to file
    cm = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.savefig(model_path + "/confusionmatrix.png")


if __name__ == "__main__":
    score_model_cm()
