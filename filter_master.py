import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np


### WARNING: there is just no way to get these imports to work
from src.datasets import Dataset


def load_info(json_path: str) -> dict:
    print("Loading dataset_info")
    with open(json_path) as dataset_json:
        return json.load(dataset_json)

dataset_info = load_info(sys.argv[1])  # run from commandline
dataset = Dataset(dataset_info['status_list'], 
                    dataset_info['base_schema'])
dataset.load_master_dataset(dataset_info['raw_master_path'],local=True)
dataset.load_orders(dataset_info['orders'])



status_list = ["Budding", "Fruiting", "Reproductive", "Flowering"]

# load all the statuses
for status in status_list:
    gt_df = dataset.load_gt(dataset_info['ground_truth'][status], [status])
    preds_df = dataset.load_preds(dataset_info['predictions'][status], [status])
    dataset.merge_preds_gt(gt_df, preds_df)
    # st.dataframe(dataview.master_df.head(10))

out_df = dataset.master_df.dropna(axis=0, how='all', subset = [f"{status} Prediction" for status in status_list])
out_df.to_csv(f"{dataset_info['name']}.csv")
