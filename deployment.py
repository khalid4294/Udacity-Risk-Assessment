"""
This script copies the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory.
"""

import os
import json
import shutil


with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
model_path = os.path.join(config["output_model_path"])


if not os.path.exists(dataset_csv_path):
    os.mkdir(dataset_csv_path)

if not os.path.exists(prod_deployment_path):
    os.mkdir(prod_deployment_path)

if not os.path.exists(model_path):
    os.mkdir(model_path)


def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    latestscore = model_path + "/latestscore.txt"
    model = model_path + "/trainedmodel.pkl"
    ingestfiles = dataset_csv_path + "/ingestedfiles.txt"
    files = [latestscore, model, ingestfiles]

    if not os.path.exists(prod_deployment_path):
        os.mkdir(prod_deployment_path)

    for file in files:
        shutil.copy2(file, prod_deployment_path)


if __name__ == "__main__":
    store_model_into_pickle()
