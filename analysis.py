"""
This is essentially a wrapper on datasets.py for making data easily accessible for graphs.

Given that all the data has already been processed, the primary work is done using threshold_single and threshold_range

"""


import json
from collections import OrderedDict
from pathlib import Path
import os
from typing import Union, List, Callable, Dict

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import boto3
import plotly.express as px

from src.views import Dataview
from src.datasets import Metric
import src.SessionState as ss
from src.ConfirmButton import cache_on_button_press
# from src.scoring import ScoringSession
from src.download_json import download_jsons

password = os.environ.get('PASSWORD')
password = "ytfc"


datasets = OrderedDict(
    [("-", None),
    ("Dataset 1", "./data/dataset_info/dataset_1.json"),
    ("Dataset 1 - 06-05-retraining Test", "./data/dataset_info/dataset_1_2021_06_05.json"),
    ("Dataset 2 - scored subset", "./data/dataset_info/dataset_2.json"),
    ("Dataset 2 - full predictions", "./data/dataset_info/dataset_2_full.json"),
    ("Dataset 2 - 02-09-21 scored subset", "./data/dataset_info/dataset_2_02_09_21.json"),
    ("Dataset 3", "./data/dataset_info/dataset_3.json"),
    ("Dataset 3 - 2021-06-15", "./data/dataset_info/dataset_3_2021_06_15.json"),
    ("Dataset 3 - 2021-07-26 (new model)", "./data/dataset_info/dataset_3_2021_07_26.json"),
    ("Dataset 3 - 2021-08-31 (new model v2)", "./data/dataset_info/dataset_3_2021_08_31.json") ]
)


mode_list = ["Homepage", "Inspection"]
DEFAULT_COLORS = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

DEFAULT_LINE_TYPES = ["solid", "dot", "dash",
                      "longdash", "dashdot", "longdashdot"]
METRIC_LINE_MAP = {"Accuracy": 'solid',
                   "Capture": 'dot'}

status_list = ['Budding', 'Flowering', 'Fruiting', 'Reproductive']

STATUS_COLOR_MAP = {"Budding": "#ff7f0e",
                    "Flowering": '#2ca02c',
                    "Fruiting": '#17becf',
                    "Reproductive": '#9467bd'}


TITLE_MAP = {"Dataset 3 - full - old_model": "Capture percentages on taxa not represented in the Training set",
            "Dataset 3 - new scoring": "Capture percentages on taxa not represented in the Training set, new model",
            "Dataset 1 - 06-05-retraining Test": "Accuracy and Capture on validation dataset"}

status = None

bucket_name = "ytfc"

s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)


def load_all():
    """
    loads all the datasets into a dictionary of dataviews, keyed by values from the above ordereddict
    """
    def process_json(json_path):

        with open(json_path, 'r') as infile:
            dataset_info = json.load(infile)

        return Dataview(dataset_info['status_list'],
                        dataset_info['base_schema'], dataset_info['master_path'])

    return {k: process_json(v) for k, v in datasets.items()}


def make_accuracy_recall_df(dv: Dataview, metrics):
    """
    Compute statistics by family

    """

    # family
    family_arr = pd.Series(dv.master_df['family'].unique(), dtype=object)

    # order
    order_list = list(dv.order_map.values()) + ["All Orders"]

    full_metrics = ['Accuracy %', 'Capture %', 'Count', 'Ground Truth Positive %', 'Ground Truth Negative %',
                    'Ground Truth Undetermined %', 'True Positive %', 'False Positive %', 'False Negative %', 'True Negative %']
    # metrics
    selected_metrics = st.multiselect(
        "Select the metrics to chart", full_metrics, ["Accuracy %", "Capture %"])

    base_metric_df, mask_df = dv.summary_pd_query(
        {"status": status_list, "family": family_arr, "order": order_list}, metrics=metrics)
    return base_metric_df


def make_accuracy_recall_curve_long(key, df: pd.DataFrame):
    df.to_csv(f"./reports/{key}_premelt.csv")
    statuses = ["Reproductive", "Flowering", "Fruiting", "Budding"]
    df.index.name = "Threshold Percent"
    long_df = df.melt(var_name="Measurement",
                      value_name="Percentage", ignore_index=False).reset_index()
    long_df.to_csv(f"./reports/{key}_melt.csv")
    return long_df


def make_accuracy_recall_curve(key: str, df: pd.DataFrame):

    statuses = ["Reproductive", "Flowering", "Fruiting", "Budding"]
    # determine if we're including accuracy or not
    print(f"DF COLUMNS: {key}")
    print(df.columns)
    accuracy_group = []
    capture_group = []
    for s in statuses:
        if f"{s} Accuracy" in df.columns:
            accuracy_group.append(f"{s} Accuracy")
        if f"{s} Capture" in df.columns:
            capture_group.append(f"{s} Capture")
    color_discrete_map = {}
    line_dash_map = {}
    # build color and line maps

    for c in df.columns:
        status, metric = c.split(' ')
        color_discrete_map[c] = STATUS_COLOR_MAP[status]
        line_dash_map[c] = METRIC_LINE_MAP[metric]

    long_df = make_accuracy_recall_curve_long(key, df)
    print(line_dash_map)
    print(color_discrete_map)
    try:
        key_title = TITLE_MAP[key]
    except KeyError:
        key_title = None
    
    fig = px.line(long_df, x='Threshold Percent', y='Percentage', color='Measurement',
                  color_discrete_map=color_discrete_map, line_dash="Measurement", line_dash_map=line_dash_map,
                  title=key_title)
    if not os.path.isdir(f"/Users/antonsquared/Projects/ytfc_image_utility/reports"):
        os.path.makedirs(
            f"/Users/antonsquared/Projects/ytfc_image_utility/reports")
    fig.write_html(f"./reports/{key}.html")


def taxa_count(df, taxa_col, truncate: Union[int, float, bool]):
    """
    computes the quantities of the dataset by taxa

    :param df: dataframe (either of scorings or raw data)
    :param taxa_col: the column to use as taxa classification (this could be o.family or sci_name or order etc.)
    :param truncate: if int: then accept this many labelled and then group the remaining taxa into "OTHER"
                     if float (and less than 1): then use this as the threshold for inclusion in the "OTHER" column
                     if bool and FALSE: no truncation
                     as a consequence: floats greater than 1 and TRUE raise errors

    """
    df['Counts'] = np.zeros(len(df))
    gp_df = df.groupby([taxa_col])[taxa_col, "Counts"].count()
    print(gp_df.head(3))
    gp_df.loc[:, "\% of total"] = gp_df.loc[:, "Counts"]/len(df)

    if isinstance(truncate, int):
        raise NotImplementedError
    elif isinstance(truncate, float):
        truncate_df = gp_df[gp_df["\% of total"] > truncate]
        
        truncate_df.loc["OTHER"] = {
            "family": np.NaN, "Counts": len(df) - truncate_df["Counts"].sum(), "\% of total": (len(df)- truncate_df["Counts"].sum())/len(df)}
        # truncate_df.drop(columns = ['order'])
        truncate_df.drop(columns=['order'], inplace=True)
        truncate_df.sort_values(by = "Counts", ascending = False, inplace=True)
        return truncate_df
    elif isinstance(truncate, bool):
        return gp_df
    else:
        raise TypeError(
            f"{truncate} is of invalid type; the value for truncate must be a integer, float, or bool")


if __name__ == "__main__":
    datasets_dict = load_all()

    # training_df = pd.read_csv("/Users/antonsquared/Projects/ytfc_image_utility/data/training_datasets/dataset_1_train.csv")
    # output_df = taxa_count(training_df, "family", 0.01)
    # output_df.drop(columns = ["family"], inplace=True)
    # output_df["\% of total"] = output_df["\% of total"] * 100
    # with open(f"dataset_1_training.tex", 'w') as out_tex:
    #     out_tex.write(output_df.to_latex(float_format="%.2f"))


    # datasets_dict.__delitem__("-")
    for k, v in datasets_dict.items():

        # accuracy/capture graphs
        try:
            df = make_accuracy_recall_df(
                v, metrics=['Accuracy %', 'Capture %'])
        except Exception as E:
            df = make_accuracy_recall_df(v, metrics=['Capture %'])
        make_accuracy_recall_curve(k, df)

        # taxa stats
        # try:
        gp_df = taxa_count(v.master_df, "order", truncate=0.02)
        gp_df.to_csv(
            f"/Users/antonsquared/Projects/ytfc_image_utility/reports/{k}_summary_by_order.csv")
        with open(f"/Users/antonsquared/Projects/ytfc_image_utility/reports/{k}.tex", 'w') as out_tex:
            out_tex.write(gp_df.to_latex(float_format="%.2f"))
            
        # except Exception as E:
        #     print(f"dataset:{k} - gp_df")
        #     print(E)

        # stats by taxa
        try:
            stats_by_taxa_df = v.stats_by_taxa("order", status_list, metrics=[
                                               'Accuracy %', 'Capture %', "Count"])
        except KeyError as E:
            print(f"dataset:{k} - stats_by_taxa")
            print(E)
            stats_by_taxa_df = v.stats_by_taxa(
                "order", status_list, metrics=['Capture %', "Count"])
        stats_by_taxa_df.to_csv(
            f"/Users/antonsquared/Projects/ytfc_image_utility/reports/{k}_stats_by_order.csv")

        # composition graphs:
