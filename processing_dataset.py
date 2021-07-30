"""
Processes datasets. Takes in the .json specifying the original scoring, ground truths for different statuses, normalizes names, binarizes scorings, and generates a singular output file 

Usage:
    processing_dataset.py -h | --help
    processing_dataset.py add <json> [-go]

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
    """
    Interface for dataset class to execute the merge and then drop rows with any N/A
    
    
    """
    args = docopt(__doc__)
    dataset_info = load_info(args['<json>'])  # run from commandline 
    dataset = Dataset(dataset_info['status_list'], 
                        dataset_info['base_schema'])
    dataset.load_master_dataset(dataset_info['raw_master_path'],local=True)
    if args['--orders']:
        dataset.load_orders(dataset_info['orders'])


    print(len(dataset.master_df.loc[dataset.master_df.index.duplicated(), :]))
    status_list = ["Budding", "Flowering", "Fruiting", "Reproductive"]

    # load all the statuses
    # for status in status_list:
    preds_df = dataset.load_preds(dataset_info['predictions']["Reproductive"], status_list)
    print(f"{len(preds_df)=}")
    if args['--ground_truth']:
        gt_df = dataset.load_gt(dataset_info['ground_truth']["Reproductive"], status_list)
        merge_gt_df = preds_df.join(gt_df, how='inner')
        dataset.merge_df_obj_id(merge_gt_df)
    else:
        dataset.merge_df_obj_id(preds_df)
    base_len = len(dataset.master_df)
    dup_len = len(dataset.master_df.loc[dataset.master_df.index.duplicated(keep=False), :])
    dup_idx_len = len(dataset.master_df.loc[dataset.master_df.index.duplicated(keep="first"), :])

        

    print(f"dedup_len: {base_len - dup_len + dup_idx_len}: {base_len}, {dup_len}, {dup_idx_len}")
    # st.dataframe(dataview.master_df.head(10))

    # TODO: dtermine why it's duplicating on every run
    print(f"{len(dataset.master_df)=}")
    dataset.master_df.to_csv('intermediate.csv')
    out_df = dataset.master_df.dropna(axis=0, how='any', subset = [f"{status} Prediction" for status in status_list])
    # print(out_df.head(10))
    print(f"{len(out_df)=}")
    out_df.drop_duplicates(inplace = True)
    print(out_df.head(3))
    print(f"{len(out_df)=}")
    out_df.to_csv(f"{dataset_info['name']}.csv")
