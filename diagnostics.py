"""
This script is used to check the health of the deployed model and the data pipeline.
"""
import pandas as pd
import timeit
import os
import json
import pickle
import subprocess


with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
model_path = os.path.join(config["prod_deployment_path"])


def model_predictions(filename):
    # read the deployed model and a test dataset, calculate predictions
    with open(model_path + "/trainedmodel.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(filename)

    X = df.drop(["exited", "corporation"], axis=1).values.reshape(
        -1, len(df.columns) - 2
    )
    y_pred = model.predict(X)
    print(f"Predictions: {y_pred}")

    return y_pred.tolist()


def dataframe_summary(filename):
    # calculate summary statistics here

    df = pd.read_csv(filename)

    summary = []
    for col in df.columns:
        if col != "corporation" and col != "exited":
            print(col)
            print(df[col].mean())
            print(df[col].median())
            print(df[col].std())
            x = [df[col].mean(), df[col].median(), df[col].std()]
            summary.append(x)

    return summary


def dataframe_nulls(filename):
    df = pd.read_csv(filename)
    nulls = df.isna().sum() / df.shape[0]

    print(f"percent of nulls: {nulls.values}")

    return nulls.tolist()


def execution_time():
    # calculate timing of training.py and ingestion.py
    start = timeit.default_timer()
    os.system("python ingestion.py")
    stop = timeit.default_timer()
    ingestion_time = stop - start
    print("ingestion time: ", ingestion_time)

    start = timeit.default_timer()
    os.system("python training.py")
    stop = timeit.default_timer()
    training_time = stop - start
    print("training time: ", training_time)

    return {"ingestion_time": ingestion_time, "training_time": training_time}


def outdated_packages_list():
    # get a list of outdated packages and their versions in requirements.txt
    outdated = subprocess.check_output(["python", "-m", "pip", "list", "--outdated"])
    outdated_nest = [i.split() for i in outdated.decode("utf-8").splitlines()][2:]
    df = pd.DataFrame(
        outdated_nest, columns=["package", "version", "latest_version", "type"]
    )

    print("Outdated packages: ")
    print(df)
    print(f"len of df: {len(df)}")

    return df


if __name__ == "__main__":
    preds_df = test_data_path + "/testdata.csv"
    summary_df = dataset_csv_path + "/finaldata.csv"
    model_predictions(preds_df)
    dataframe_summary(summary_df)
    execution_time()
    outdated_packages_list()
