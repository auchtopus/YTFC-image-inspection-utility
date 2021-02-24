## TODO: - update the files with orders using process_dataset.py
## TODO: add db support
## TODO: prevent intermediate compilation of the middle steps

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

from src.views import Dataview
from src.datasets import Metric
import src.SessionState as ss
from src.ConfirmButton import cache_on_button_press
from src.scoring import ScoringSession
from src.download_json import download_jsons

password = os.environ.get('PASSWORD')
password = "ytfc" 

datasets = OrderedDict(
    [("-", None),
    ("Dataset 1", "./data/dataset_info/dataset_1.json"),
    ("Dataset 2 - scored subset", "./data/dataset_info/dataset_2.json"),
    ("Dataset 2 - full predictions", "./data/dataset_info/dataset_2_full.json"),
    ("Dataset 2 - 02-09-21 scored subset", "./data/dataset_info/dataset_2_02_09_21.json"),
    ("Dataset 3", "./data/dataset_info/dataset_3.json")]
)

load_mask = OrderedDict(
    [("-", None),
    ("Dataset 1", False),
    ("Dataset 2 - scored subset", False),
    ("Dataset 2 - full predictions", False),
    ("Dataset 2 - 02-09-21 scored subset", True),
    ("Dataset 3", False)]
)

mode_list = ["Homepage", "Inspection", "Scoring"]
scoring_sets = OrderedDict(
    [("-", None),
     ("test", "./data/scoring_info/02_06_2021.json"),
     ("Scoring: Feb 6, 2021", "./data/scoring_info/02_06_2021.json")]
)

status_list = ['Budding', 'Flowering', 'Fruiting', 'Reproductive']

status = None

bucket_name = "ytfc"

s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)



# now we serialize everything

@st.cache
def load_info(json_path: str) -> dict:
    print("Loading dataset_info")
    with open(json_path) as dataset_json:
        return json.load(dataset_json)


@st.cache(allow_output_mutation=True)
def perm_obj(obj):
    return obj

def homepage():
    st.write("Homepage!")
    st.write("choose a mode at left")

def inspection():
    
    loaded_dict = perm_obj({})
    dataset_name = st.sidebar.selectbox("Choose a Dataset", list(datasets.keys()), 0) # 
    # returns the second element of the tuple with first value being the selection

    dataset_json = datasets[dataset_name]
    req_load = load_mask[dataset_name]

    if dataset_name == "-":
        st.write("Welcome!")
        intro_mkd = Path("./assets/introduction.md").read_text()
        st.markdown(intro_mkd)
    else:
        print(loaded_dict.keys())
        if dataset_name not in loaded_dict:   


            dataset_info = load_info(dataset_json)

            
            with st.spinner('Loading data...'):
                # preprocessing is no longer done on client side
                dataview = Dataview(dataset_info['status_list'], 
                                    dataset_info['base_schema'], dataset_info['master_path'])

            loaded_dict[dataset_name] = dataview
                
        
        status_select = st.multiselect("Choose a phenological status to analyze", status_list)

        if len(status_select) == 0:
            st.write("Select phenological statuses to get started")
        else:
            # family
            family_arr = pd.Series(loaded_dict[dataset_name].master_df['family'].unique(),dtype=object)
            family_arr = list(family_arr.append(pd.Series(["All Families"]), ignore_index = True))
            families = st.multiselect("Select the families to analyze", family_arr, [])


            # order
            order_list = list(loaded_dict[dataset_name].order_map.values()) + ["All Orders"]
            orders = st.multiselect("Select the orders to analyze, or select `All Orders`", order_list, ["All Orders"])
            full_metrics = ['Accuracy %', 'Capture %', 'Count', 'Ground Truth Positive %', 'Ground Truth Negative %', 'Ground Truth Undetermined %', 'True Positive %', 'False Positive %', 'False Negative %', 'True Negative %']
            # metrics
            selected_metrics = st.multiselect("Select the metrics to chart", full_metrics, ["Accuracy %", "Capture %"])

            
            query_dict = {"status": status_select,
                        "family": families,
                        "order": orders}

            base_metric_df, mask_df = loaded_dict[dataset_name].summary_pd_query(query_dict, selected_metrics)

            st.line_chart(base_metric_df)
            # st.dataframe(mask_df)
            # st.dataframe(base_metric_df)

            # process individual samples
            lower_bound = st.slider("Select the lower confidence bound", 0.5, 1.0 , 0.9, 0.01)

            sample_df = loaded_dict[dataset_name].sample_pd_query(mask_df, status_select, lower_bound)

            st.dataframe(sample_df)

            # image loading
            image_name = st.selectbox("Image to display", sample_df.index)

            st.dataframe(sample_df.loc[image_name])
            image_url = loaded_dict[dataset_name].master_df.loc[image_name, "url"]
            response = requests.get(image_url)
            sample_image = Image.open(BytesIO(response.content))
            st.image(sample_image, caption = f"sample: {image_name}", use_column_width=True)

def authentication():
    @cache_on_button_press('Authenticate')
    def authenticate(password):
        return password == os.environ.get("YTFC_PASSWORD")


    password = st.text_input('Password')
    session_key = st.selectbox('Session Key', ['-', '02_09_2021'], 0)

    if authenticate(password):
        scoring_session = ScoringSession(session_key)
        scoring_session.score()

    else:
        st.error('The username or password you have entered is invalid.')

def scoring():
    authentication()


def run():
    mode = st.sidebar.selectbox("Chose a Mode", mode_list, 0)
    if mode == "Homepage":
        homepage()

    if mode == "Inspection":
        inspection()
    
    if mode == "Scoring":
        scoring()


if __name__ == "__main__":
    run()