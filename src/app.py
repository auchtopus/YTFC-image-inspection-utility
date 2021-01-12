import json
from collections import OrderedDict
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from views import Dataview



DATASETS = OrderedDict(
    [("-", None),
    ("Dataset 1", "./data/dataset_info/dataset_1.json"),
    ("Dataset 2", "./data/dataset_info/dataset_2.json"),
    ("Dataset 3", "./data/dataset_info/dataset_3.json")]
)



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
    dataset_name = st.sidebar.selectbox("Choose a Dataset", list(DATASETS.keys()), 0) # 
    # returns the second element of the tuple with first value being the selection

    dataset_json = DATASETS[dataset_name]

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
            for status in ['Budding', 'Flowering', 'Fruiting', 'Reproductive']:
                dataview.load_prediction_set([status], dataset_info['ground_truth'][status], dataset_info['predictions'][status])
                # st.dataframe(dataview.master_df.head(10))

        statuses = ["All", "Budding", "Flowering", "Fruiting", "Reproductive"]
        
        active_status = st.selectbox("Choose a phenological status to analyze", statuses, 0)

        # st.dataframe(loaded_dict[dataset_name].master_df.head(20))

        # st.write(loaded_dict[dataset_name].master_df.columns.values)


        family_arr = pd.Series(loaded_dict[dataset_name].master_df['family'].unique(),dtype=object)
        family_arr = list(family_arr.append(pd.Series(["All Families"]), ignore_index = True))
        families = st.multiselect("Select the families to analyze", family_arr, ["All Families"])

        order_list = list(loaded_dict[dataset_name].order_map.values()) + ["All Orders"]
        orders = st.multiselect("Select the orders to analyze", order_list, ["All Orders"])

        lower_bound = st.slider("Select the lower confidence bound", 0.5, 1.0 , 0.9, 0.01)

        
        query_dict = {"status": active_status,
                      "family": families,
                      "order": orders,
                      "confidence": lower_bound}


        loaded_dict[dataset_name].summary_pd_query(query_dict)




        






if __name__ == "__main__":
    run()