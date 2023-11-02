"""
script to call all API endpoints and store the responses
"""

import requests
import json
import os

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

with open("config.json", "r") as f:
    config = json.load(f)

model_path = os.path.join(config["output_model_path"])
dataset_csv_path = os.path.join(config["output_folder_path"])
dataset = "finaldata.csv"
test_data_path = os.path.join(config["test_data_path"])
test_dataset = "testdata.csv"

# Call each API endpoint and store the responses
# pass test data to prediction endpoint
response1 = requests.post(
    URL + "prediction", params={"filename": test_data_path + "/" + test_dataset}
)
print(response1.text)

response2 = requests.get(URL + "scoring")
print(response2.text)

response3 = requests.get(
    URL + "summarystats", params={"filename": dataset_csv_path + "/" + dataset}
)
print(response3.text)

response4 = requests.get(
    URL + "diagnostics", params={"filename": dataset_csv_path + "/" + dataset}
)
print(response4.text)

# combine all API responses
responses = {
    "prediction": response1.text,
    "scoring": response2.text,
    "summarystats": response3.text,
    "diagnostics": response4.text,
}

with open(model_path + "/apireturns.json", "w") as f:
    json.dump(responses, f)
