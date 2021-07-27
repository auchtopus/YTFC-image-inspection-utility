"""
This is essentially a wrapper on datasets.py for making data easily accessible for graphs.

Given that all the data has already been processed, the primary work is done using threshold_single and threshold_range

"""


import json
from collections import OrderedDict
from pathlib import Path
import os


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
     ("Dataset 1 - 06-05-retraining Test",
      "./data/dataset_info/dataset_1_2021_06_05.json"),
     ("Dataset 2 - scored subset", "./data/dataset_info/dataset_2.json"),
     ("Dataset 2 - full predictions", "./data/dataset_info/dataset_2_full.json"),
     ("Dataset 2 - 02-09-21 scored subset",
      "./data/dataset_info/dataset_2_02_09_21.json"),
     ("Dataset 3", "./data/dataset_info/dataset_3.json"),
     ("Dataset 3 - full", "./data/dataset_info/dataset_3_2021_06_15.json")]
)


mode_list = ["Homepage", "Inspection"]


status_list = ['Budding', 'Flowering', 'Fruiting', 'Reproductive']

status = None

bucket_name = "ytfc"

s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)


def load_all():
    """
    loads all the datasets into a dictionary of dataviews, keyed by values from the above ordereddict
    """
    def process_json(json_path):
        try:
            with open(json_path, 'r') as infile:
                dataset_info = json.load(infile)

            return Dataview(dataset_info['status_list'],
                            dataset_info['base_schema'], dataset_info['master_path'])
        except TypeError:
            return None

    return {k: process_json(v) for k,v in datasets.items()}

def make_accuracy_recall_df(dv: Dataview, metrics):

    # famil
    family_arr = pd.Series(dv.master_df['family'].unique(),dtype=object)


    # order
    order_list = list(dv.order_map.values()) + ["All Orders"]

    full_metrics = ['Accuracy %', 'Capture %', 'Count', 'Ground Truth Positive %', 'Ground Truth Negative %', 'Ground Truth Undetermined %', 'True Positive %', 'False Positive %', 'False Negative %', 'True Negative %']
    # metrics
    selected_metrics = st.multiselect("Select the metrics to chart", full_metrics, ["Accuracy %", "Capture %"])


    base_metric_df, mask_df = dv.summary_pd_query({"status": status_list, "family": family_arr, "order": order_list}, metrics = metrics)
    return base_metric_df


def make_accuracy_recall_curve(key: str, df: pd.DataFrame):
    fig = px.line(df)
    fig.write_html(f"{key}.html")


if __name__ == "__main__":
    datasets_dict = load_all()
    datasets_dict.__delitem__("-")
    for k, v in datasets_dict.items():


        # accuracy/capture graphs
        try:
            df = make_accuracy_recall_df(v, metrics = ['Accuracy %', 'Capture %'])
        except Exception as E:
            df = make_accuracy_recall_df(v, metrics = [ 'Capture %'])
        make_accuracy_recall_curve(k, df)


        # composition graphs:
        