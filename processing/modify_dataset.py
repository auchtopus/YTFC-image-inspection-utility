"""
Script to update datasets

1. update certain fields
2. delete certain entries (basically just wraps df.drop)


"""

import sys

import pandas as pd
import numpy as np



def update_field(target_df: str, update_df: str, update_map: dict):
    """
    target_df: the target df to update
    update_df: the df with information with which to update
    update_map: normalize the column names of the update_df with the target_df standard

    """
    # identify by the catalog_number
    
    update_df = update_df.rename(columns = update_map)
    print([v for i,v in update_map.items()])
    update_df = update_df[[v for i,v in update_map.items()]]
    target_df.update(update_df)
    join_df = pd.merge(target_df, update_df, how='inner', left_index=True,right_index=True)
    print(join_df)
    target_df.to_csv('updated_family.csv')


def delete_entries(target_df, update_list):
    target_df.drop(update_list, inplace=True, errors = 'ignore')
    target_df.to_csv('updated.csv')


if __name__ == "__main__":

    target_path = sys.argv[1]
    # update_path =sys.argv[2]
    delete_path = sys.argv[2]

    target_df = pd.read_csv(target_path, index_col = 'catalog_number')
    # update_df = pd.read_csv(update_path, index_col = 'o.catalogNumber')
    delete_df = pd.read_csv(delete_path, index_col = 'catalogNumber')
    
    delete_list = delete_df.index.values


    # update_field(target_df, update_df, {"accepted_family": "family", "o.scientificName": "scientific_name"})
    delete_entries(target_df, delete_list)


