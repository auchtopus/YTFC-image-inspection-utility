import json
import os
import sys


import pandas as pd
import numpy as np
import boto3
import botocore


bucket_name = "ytfc"

s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)


def parse(status):
    if status == "true" or status == True:
        return True
    elif status == "false" or status == False:
        return False
    else:
        return np.nan


def parse_original(status):
    if status == "true" or status:
        return True
    elif status == "false" or not status:
        return False
    else:
        return np.nan

def download_jsons(session_key, save_all = False):
    print(session_key)
    bucket.download_file(f"scoring_info/{session_key}/{session_key}_scoring.csv", f"{session_key}_scoring.csv")
    manifest_df = pd.read_csv(f"{session_key}_scoring.csv", index_col="catalog_number")
    item_list = manifest_df.index.values
    for item in item_list:
        try:
            # print(f"item: {item} exists")
            if save_all:
                bucket.download_file(f"scoring_info/{session_key}/{item}.json", f"{item}.json")
                with open(f"{item}.json", 'r') as in_json:
                    item_info = json.load(in_json)
            else:
                bucket.download_file(f"scoring_info/{session_key}/{item}.json", f"item.json")
                with open(f"item.json", 'r') as in_json:
                    item_info = json.load(in_json)

            for status in ['Reproductive', 'Budding', 'Fruiting', 'Flowering']:
                manifest_df.loc[item_info['catalog_number'], f"{status} Ground Truth"] = parse(item_info[status])
                print(parse(item_info[status]))


        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print(f"{item} does not exist.")
        
    manifest_df.to_csv(f'./data/filter_master/{session_key}_bad_scoring.csv')



if __name__ == "__main__":
    download_jsons(sys.argv[1])