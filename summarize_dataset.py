"""
Generate df's summarizing datasets. Sets a 0.96 confidence interval, 

    # result_df['Accuracy %'] = result_df.apply(lambda row: f"{row['Accuracy %'] * 100:0.2f}", axis = 1)
    # result_df['Capture %'] = result_df.apply(lambda row: f"{row['Capture %'] * 100:0.2f}", axis = 1)
    # result_df['Composition %'] = result_df.apply(lambda row: f"{row['Composition %'] * 100:0.2f}", axis = 1)
"""


import pandas as pd
from typing import List
from src.views import Dataview
import sys

def summarize_dataset_metrics(status: str, path: str, c_level: float, metrics: List[str]):
    """
    computes metrics defined in metrics
    status: reproductive status
    path: path to full scoring
    c_level: confidence level, float
    metrics: List of metrics to compute

    """
    dataset = Dataview(['Reproductive', 'Fruiting', 'Flowering', 'Budding'], {}, path)
    family_set = set(dataset.master_df['family'])
    full_dict = {}
    for family in family_set:
        summary_df, _ = dataset.summary_pd_query({"status": [status],
            "family": [family], "order": [None]}, metrics + ["Count"])
        full_dict[family] = { metric: summary_df.loc[c_level, f'{status} {metric[:-2]}'] for metric in metrics}
        full_dict[family]['Number of Samples'] = summary_df.loc[0.5, f'{status} Number of Samples']



    ## aggregate
    summary_df, _ = dataset.summary_pd_query({"status": [status],
        "family": family_set, "order": [None]}, metrics+ ["Count"])
    full_dict['Total'] = { metric: summary_df.loc[c_level, f'{status} {metric[:-2]}'] for metric in metrics}
    full_dict['Total']['Number of Samples'] = summary_df.loc[0.5, f'{status} Number of Samples']


    result_df = pd.DataFrame.from_dict(full_dict, orient='index')

    result_df.sort_index(inplace=True)

    result_df['Composition %'] = result_df.apply(lambda row: row['Number of Samples']/full_dict['Total']['Number of Samples'], axis = 1)

    # result_df['Accuracy %'] = result_df.apply(lambda row: f"{row['Accuracy %'] * 100:0.2f}", axis = 1)
    # result_df['Capture %'] = result_df.apply(lambda row: f"{row['Capture %'] * 100:0.2f}", axis = 1)
    # result_df['Composition %'] = result_df.apply(lambda row: f"{row['Composition %'] * 100:0.2f}", axis = 1)

    return result_df

# def compare(status, sample_file, main_file, c_level):



# summarize_dataset("Reproductive", "/Users/antonsquared/Projects/ytfc_image_utility/data/filter_master/dataset_2.csv", 0.95)


if __name__ == "__main__":
    # summarize_dataset(sys.argv[1], sys.argv[2], sys.argv[3])
    for status in ["Reproductive", "Flowering", "Fruiting", "Budding"]:
        print(summarize_dataset_metrics(status, "/Users/antonsquared/Projects/ytfc_image_utility/data/filter_master/dataset_1.csv", 0.95, ["Capture %", "Accuracy %"]))




