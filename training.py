"""
This script trains a logistic regression model on the data in the finaldata.csv file.
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import pickle
import os
import json

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
model_path = os.path.join(config["output_model_path"])


def train_model():
    df = pd.read_csv(dataset_csv_path + "/finaldata.csv")
    X = df.drop(["exited", "corporation"], axis=1).values.reshape(
        -1, len(df.columns) - 2
    )
    y = df["exited"].values.reshape(-1, 1).ravel()

    # use this logistic regression for training
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",  # was warn, but deprecated
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    # fit the logistic regression to your data
    model.fit(X, y)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model_file = "trainedmodel.pkl"
    pickle.dump(model, open(model_path + "/" + model_file, "wb"))


if __name__ == "__main__":
    train_model()
