import json
from collections import OrderedDict
from pathlib import Path


import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO

from views import Dataview
from datasets import Metric


datasets = OrderedDict(
    [("-", None),
    ("Dataset 1", "./data/dataset_info/dataset_1.json"),
    ("Dataset 2", "./data/dataset_info/dataset_2.json"),
    ("Dataset 3", "./data/dataset_info/dataset_3.json")]
)

status_list = ['Budding', 'Flowering', 'Fruiting', 'Reproductive']

status = None



# now we serialize everything

@st.cache
def load_info(json_path: str) -> dict:
    print("Loading dataset_info")
    with open(json_path) as dataset_json:
        return json.load(dataset_json)

@st.cache(allow_output_mutation=True)
def loaded_dict_wrapper():
    return {}


def run():
    loaded_dict = loaded_dict_wrapper()
    dataset_name = st.sidebar.selectbox("Choose a Dataset", list(datasets.keys()), 0) # 
    # returns the second element of the tuple with first value being the selection

    dataset_json = datasets[dataset_name]

    dataview_1, dataview_2, dataview_3 = None, None, None

    if dataset_name == "-":
        st.write("Welcome!")
        intro_mkd = Path("./assets/introduction.md").read_text()
        st.markdown(intro_mkd)

    else:
        print(loaded_dict.keys())
        if dataset_name not in loaded_dict:
            dataset_info = load_info(dataset_json)
            with st.spinner('Loading data...'):
                dataview = Dataview(dataset_info['status_list'], 
                                    dataset_info['base_schema'],
                                    dataset_info['master_dataset_path'],
                                    dataset_info['orders'])
                
                loaded_dict[dataset_name] = dataview

            # load all the statuses
            for status in status_list:
                dataview.load_prediction_set([status], dataset_info['ground_truth'][status], dataset_info['predictions'][status])
                # st.dataframe(dataview.master_df.head(10))

        
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
            full_metrics = ['Accuracy %', 'Capture %', 'F1 Score', 'Precision', 'Recall', 'Ground Truth Positive %', 'Ground Truth Negative %', 'Ground Truth Undetermined %', 'True Positive %', 'False Positive %', 'False Negative %', 'True Negative %']
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
                
            







if __name__ == "__main__":
    run()