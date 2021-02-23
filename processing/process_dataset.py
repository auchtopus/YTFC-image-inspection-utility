"""
Processes datasets.

Usage:
    process_dataset.py -h | --help
    process_dataset.py add <json> [-go]

Options:
    -h --help                  Show help
    -g --ground_truth          Include ground truth
    -o --orders                Include orders

"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from docopt import docopt



from src.datasets import Dataset


def load_info(json_path: str) -> dict:
    print("Loading dataset_info")
    with open(json_path) as dataset_json:
        return json.load(dataset_json)


if __name__ == "__main__":
    args = docopt(__doc__)
    dataset_info = load_info(args['<json>'])  # run from commandline 
    dataset = Dataset(dataset_info['status_list'], 
                        dataset_info['base_schema'])
    dataset.load_master_dataset(dataset_info['raw_master_path'],local=True)
    if args['--orders']:
        dataset.load_orders(dataset_info['orders'])



    status_list = ["Budding", "Flowering", "Fruiting", "Reproductive"]

    # load all the statuses
    for status in status_list:
        preds_df = dataset.load_preds(dataset_info['predictions'][status], [status])
        if args['--ground_truth']:
            gt_df = dataset.load_gt(dataset_info['ground_truth'][status], [status])
            merge_gt_df = preds_df.join(gt_df, how='inner')
            dataset.merge_df(merge_gt_df)
        else:
            dataset.merge_df(preds_df)
        # st.dataframe(dataview.master_df.head(10))

    out_df = dataset.master_df.dropna(axis=0, how='all', subset = [f"{status} Prediction" for status in status_list])
    out_df.to_csv(f"{dataset_info['name']}.csv")
