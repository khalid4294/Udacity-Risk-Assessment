"""
This script runs the full process of the ML pipeline. It checks for new data, checks for drift, and if drift is detected, retrains the model and runs the full pipeline.
"""

import ast
import os
import json
import scoring


with open("config.json") as f:
    config = json.load(f)


def has_new_data(config):
    with open(config["prod_deployment_path"] + "/ingestedfiles.txt") as f:
        files = ast.literal_eval(f.read())

    new_files = os.listdir(config["input_folder_path"])
    new_files = [file for file in new_files if file.endswith(".csv")]

    has_new = False
    for file in new_files:
        if file not in files:
            has_new = True
            break

    return has_new


def check_drift(config):
    print("new data detected")
    print("ingestion running ...")
    os.system("python ingestion.py")

    with open(config["prod_deployment_path"] + "/latestscore.txt") as f:
        prod_score = float(f.read())

    new_score = scoring.score_model(
        model_path=config["prod_deployment_path"],
        data_path=config["output_folder_path"] + "/finaldata.csv",
    )

    print(f"old score: {prod_score}")
    print(f"new score: {new_score}")
    if new_score < (prod_score * 0.95) or new_score > (prod_score * 1.05):
        return True
    else:
        return False


if __name__ == "__main__":
    print("checking for new data")
    has_new = has_new_data(config)
    print(f"has new data: {has_new}")

    if has_new:
        print("checking for drift")
        retrain = check_drift(config)
        print(f"needs retrain: {retrain}")
    else:
        retrain = False

    if retrain:
        print("Model drift detected. Running full pipeline.")

        print("training running")
        os.system("python training.py")

        print("deployment running")
        os.system("python deployment.py")

        print("diagnostics running")
        os.system("python diagnostics.py")

        print("reporting running")
        os.system("python reporting.py")
