import pandas as pd
import numpy as np
import streamlit as st
from collections import defaultdict

from datasets import Dataset

NEVP_2019_09_10_bindings = {"catalog_number":  "catalogNumber", # this functions as the primary key
                            "url": "originalurl",
                            "sci_name": "scientificName",
                            "family": "family",
                            "order": None,
                            "inst": "institutionCode",
                            "object_id": "occid",
                            "state": "stateProvince",
                            "event_date": "eventDate",
                            "year": "year",
                            "county": "county",
                            "municipality": "municipality",
                            "locality": "locality",
                            "latitude": "decimalLatitude",
                            "longitue": "decimalLongitude"}


class CacheDataset(Dataset):
    """
    st.cache version of datasets.py
    """

    def __init__(self, label_map):
        super().__init__(label_map)

    @st.cache
    def load_master_dataset(self, csv_path):
        return super().load_master_dataset(csv_path)

    @st.cache
    def load_gt(self, gt_csv_path, status_list):
        return super().load_gt(gt_csv_path, status_list)

    @st.cache
    def load_orders(self, order_csv_path, head_label=True):
        return super().load_orders(order_csv_path, head_label=head_label)

    @st.cache
    def load_preds(self, pred_csv_path, status_list, binarized=False):
        return super().load_preds(pred_csv_path, status_list, binarized=binarized)

    @st.cache
    def merge_preds_gt(self, preds_df, gt_df):
        return super().merge_preds_gt(preds_df, gt_df)
    

class Dataview(CacheDataset): #TODO determine where to put queries

    def __init__(self, status_list, label_map, master_dataset_path, order_csv_path):
        super().__init__(label_map)
        super().load_master_dataset(master_dataset_path) # this gives us self.master_df
        super().load_orders(order_csv_path)
        self.fields = self.master_df.columns.values
        
        

    def load_prediction_set(self, status_list, gt_csv_path, pred_csv_path):
        """
        There can be multiple gt/pred combinations appended to a single master list.
        """

        gt_df =super().load_gt(gt_csv_path, status_list)
        preds_df = super().load_preds(pred_csv_path, status_list)
        super().merge_preds_gt(preds_df, gt_df)
        self.fields = self.master_df.columns.values
        


    def summary_pd_query(self, query): # gets numbers, not samples
        """
        Query format: {status : str,
             family: [vals],
             order: [vals], 
             {status} Prediction: True/False,
             {status} Prediction Confidence: np.linspace,
             metrics: ['Accuracy', 'F1-score','Precision', 'Recall',True Positive', 'False Positive', 'True Negative', 'False Negative']}
        """
        if f"{query['status']} Prediction" not in self.fields:
            raise IndexError(f"No predictions for status: {query['status']}")


        # work through the entries to build the query. Would sql be faster?
        mask = pd.Series([True] * len(self.master_df))
        if query['family'] != '-':
            mask = mask & (self.master_df["family"].isin(query["family"]))
        if query['order'] != '-':
            mask = mask & (self.master_df["family"].isin(query["family"]))
        if query[f"{query['status']} Prediction"] != '-':
            mask = mask & (self.master_df[f"{query['status']} Prediction"] == f"{query['status']} Prediction")
        if query[f"{query['status']} Prediction Confidence"] != '-':
            mask = mask & (self.master_df[f"{query['status']} Prediction Confidence"] > query[f"{query['status']} Prediction Confidence"])
        if query[f"{query['status']} Prediction"]:
            def filter_conf_state(row, pred, gt):
                return row[f"{query['status']} Prediction"] == pred and row[f"{query['status']} Ground Truth" ] == gt
            state_dict = {"True Positive": [True, True],
                          "False Positive": [True, False],
                          "True Negative": [False, False],
                          "False Negative": [False, True]}
            for confusion_state in query["confusion_state"]:
                state = state_dict[confusion_state]
                mask = mask & (self.master_df.apply(filter_conf_state, axis = 1, pred = state[0], gt = state[1]))







    def sample_pd_query(self, query: defaultdict, col_list): # gets individual samples
        """
        This is at least 5x slower than using sql, but using sql adds overheard for going from pandas to sql because the logic cannot be moved out of pandas



        Query format: (default_dict)
            {status : str,
             family: [vals],
             order: [vals], 
             {status} Prediction: True/False,
             {status} Prediction Confidence: float,
             confusion_state: ['True Positive', 'False Positive', 'True Negative', 'False Negative']}
        """
    
        # a status is necessary to see meaningful information for samples
        if f"{query['status']} Prediction" not in self.fields:
            raise IndexError(f"No predictions for status: {query['status']}")


        # work through the entries to build the query. Would sql be faster?
        mask = pd.Series([True] * len(self.master_df))
        if query['family'] != '-':
            mask = mask & (self.master_df["family"].isin(query["family"]))
        if query['order'] != '-':
            mask = mask & (self.master_df["family"].isin(query["family"]))
        if query[f"{query['status']} Prediction"] != '-':
            mask = mask & (self.master_df[f"{query['status']} Prediction"] == f"{query['status']} Prediction")
        if query[f"{query['status']} Prediction Confidence"] != '-':
            mask = mask & (self.master_df[f"{query['status']} Prediction Confidence"] > query[f"{query['status']} Prediction Confidence"])
        if query[f"{query['status']} Prediction"]:
            def filter_conf_state(row, pred, gt):
                return row[f"{query['status']} Prediction"] == pred and row[f"{query['status']} Ground Truth" ] == gt
            state_dict = {"True Positive": [True, True],
                          "False Positive": [True, False],
                          "True Negative": [False, False],
                          "False Negative": [False, True]}
            for confusion_state in query["confusion_state"]:
                state = state_dict[confusion_state]
                mask = mask & (self.master_df.apply(filter_conf_state, axis = 1, pred = state[0], gt = state[1]))
    
        mask_df = self.master_df[mask]
        return mask_df[col_list]
        



        

        

    
    



        

