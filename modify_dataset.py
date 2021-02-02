import sys

import pandas as pd
import numpy as np



from src.datasets import Dataset

def update_field(target_df: str, update_df: str, update_map: dict):
    # identify by the catalog_number
    
    update_df = update_df.rename(columns = update_map)
    print([v for i,v in update_map.items()])
    update_df = update_df[[v for i,v in update_map.items()]]
    target_df.update(update_df)
    target_df.to_csv('updated.csv')



if __name__ == "__main__":

    target_path = sys.argv[1]
    update_path =sys.argv[2]

    target_df = pd.read_csv(target_path, index_col = 'catalog_number')
    update_df = pd.read_csv(update_path, index_col = 'catalogNumber')
    


    update_field(target_df, update_df, {"updated_family": "family", "scientificName": "scientific_name"})

