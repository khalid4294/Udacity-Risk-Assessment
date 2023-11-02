"""
This script is used to ingest data from multiple files and merge them into a single file.
"""

import pandas as pd
import os
import json


with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]


def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file

    file_names = os.listdir(input_folder_path)
    file_names = [file for file in file_names if file.endswith(".csv")]

    for idx, file in enumerate(file_names):
        df_temp = pd.read_csv(os.path.join(input_folder_path, file))

        if idx == 0:
            df = df_temp
        else:
            df = pd.concat([df, df_temp], axis=0, ignore_index=True).drop_duplicates()

    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    df.to_csv(output_folder_path + "/finaldata.csv", index=False)

    with open(output_folder_path + "/ingestedfiles.txt", "w") as f:
        f.write(str(file_names))


if __name__ == "__main__":
    merge_multiple_dataframe()
