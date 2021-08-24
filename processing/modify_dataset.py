"""
Script to update datasets

1. update certain fields
2. delete certain entries (basically just wraps df.drop)

May need to manually adjust indexes


Usage:
    modify_dataset.py delete <target_file> <delete_file> <output_file_name>
    modify_dataset.py update <target_file> <update_file> <output_file_name>
    modify_dataset.py test-delete <target_file> <delete_file>
    modify_dataset.py test-update <target_file> <update_file>

"""
from typing import List
import sys

import pandas as pd
import numpy as np
import docopt


def update_field(target_df: str, update_df: str, update_map: dict):
    """
    target_df: the target df to c
    update_df: the df with information with which to update
    update_map: normalize the column names of the update_df with the target_df standard

    """
    # identify by the catalog_number

    update_df = update_df.rename(columns=update_map)
    print([v for i, v in update_map.items()])
    update_df = update_df[[v for i, v in update_map.items()]]

    # dedup inputs
    print(len(update_df), len(target_df))
    update_df = update_df.loc[~update_df.index.duplicated(keep='first')]
    target_df = target_df.loc[~target_df.index.duplicated(keep="first")]
    print(len(update_df), len(target_df))
    print(len(target_df.index.duplicated()))
    print(len(update_df.index.duplicated()))
    # update_df = update_df.loc[update_df_new_index, :]
    # print(target_df.index.duplicated())
    # 2021-07-25: I think there's a bug in the update code
    # print(update_df.columns)
    # print(update_df.reindex_like(target_df))
    # target_df.update(update_df, errors = 'ignore')
    # join_df = pd.merge(target_df, update_df, how='left', left_index=True,right_index=True)
    # print(join_df)

    # update_df.index.drop_duplicates
    target_df.loc[target_df.index.duplicated(), :].to_csv("duplicates.csv")

    # manually update:

    target_df.update(update_df, overwrite=True, errors='ignore')

    return target_df


def delete_entries(target_df, update_list):
    pruned_df = target_df.drop(update_list, errors='ignore')
    pruned_df.index.name = "o.CNH_id"
    return pruned_df


def test_deleted_entries(target_df: pd.DataFrame, delete_list: List[str]):
    """
    Verifies that no elements of delete_list exist in target_df

    """
    assert len(set(target_df.index) & set(delete_list)) == 0
    print("Assertion passed!")

if __name__ == "__main__":
    args = docopt.docopt(__doc__)

    if args["delete"]:
        target_path = args["<target_file>"]
        # update_path =sys.argv[2]
        delete_path = args["<delete_file>"]

        target_df = pd.read_csv(target_path, index_col='object_id')
        # update_df = pd.read_csv(update_path, index_col = 'o.catalogNumber')
        delete_df = pd.read_csv(delete_path, index_col='occid')

        delete_list = delete_df.index.values

        # update_field(target_df, update_df, {"accepted_family": "family", "o.scientificName": "scientific_name"})
        output_df = delete_entries(target_df, delete_list)
        test_deleted_entries(output_df, delete_list)

    elif args["update"]:
        target_path = args["<target_file>"]
        update_path = args["<update_file>"]

        target_df = pd.read_csv(target_path, index_col='object_id')
        update_df = pd.read_csv(update_path, index_col='o.CNH_id')
        # delete_df = pd.read_csv(delete_path, index_col = 'catalogNumber')

        # delete_list = delete_df.index.values

        output_df = update_field(target_df, update_df, {
                                 "accepted_family": "o.family", "o.scientificName": "o.scientificName"})
        
        output_df.to_csv(f"{args['<output_file_name>']}")
        
    elif args["test-delete"]:
        target_path = args["<target_file>"]
        # update_path =sys.argv[2]
        delete_path = args["<delete_file>"]

        target_df = pd.read_csv(target_path, index_col='o.CNH_id')
        # update_df = pd.read_csv(update_path, index_col = 'o.catalogNumber')
        delete_df = pd.read_csv(delete_path, index_col='occid')

        delete_list = delete_df.index.values

        # update_field(target_df, update_df, {"accepted_family": "family", "o.scientificName": "scientific_name"})
        # output_df = delete_entries(target_df, delete_list)
        test_deleted_entries(target_df, delete_list)

    elif args["test-rename"]:
        target_path = args["<target_file>"]
        update_path = args["<update_file>"]

        target_df = pd.read_csv(target_path, index_col='object_id')
        update_df = pd.read_csv(update_path, index_col='o.CNH_id')
        # delete_df = pd.read_csv(delete_path, index_col = 'catalogNumber')

        # delete_list = delete_df.index.values

        output_df = update_field(target_df, update_df, {
                                 "accepted_family": "o.family", "o.scientificName": "o.scientificName"})
        
        output_df.to_csv(f"{args['<output_file_name>']}")
