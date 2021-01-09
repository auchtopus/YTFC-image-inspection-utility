import pandas as pd
import numpy as np
from abc import ABC




"""
Each dataset is related to an original master_dataset. One Dataset object can have multiple scorings associated with it, but only one master_dataset. 

This module makes no use of streamlit utilities
"""
class Dataset:

    def __init__(self, label_map: dict = {}):
        self.master_df = None

        # we will rename every column to these standardized names
        self.label_map= {"catalog_number":  None, # this functions as the primary key
                         "url": None,
                         "sci_name": None,
                         "family": None,
                         "order": None,
                         "inst": None,
                         "catalog_id": None,
                         "state": None,
                         "date": None,
                         "county": None,
                         "municipality": None,
                         "locality": None,
                         "latitude": None,
                         "longitue": None}
        self.label_map.update(label_map)
        self.order_map = {}
        self.status_list = ['Budding', 'Flowering', 'Fruiting' ,'Reproductive']

    def load_master_dataset(self, csv_path):
        self.master_df = pd.read_csv(csv_path)
        self.master_df.rename(columns = {v:k for k,v in self.label_map.items()},inplace=True)
        # deduping because we use our own ground truth, and only need one version of every file
        self.master_df.drop_duplicates(subset = ['catalog_number'], keep='first', inplace=True)
        self.master_df.set_index('catalog_number', inplace = True)

    

    def load_orders(self, order_csv_path, head_label = True):
        """
        populates the 'order' field of self.master_df using a csv mapping families to orders

        Arguments:
            order_csv_path (str): path to the csv storing orders
            head_label (str): whether the first row is a label

        order_csv_path schema:
        Family: (1st column): str of family
        Order:  (2nd column): str of order


        """
        if head_label:
            mapping_df = pd.read_csv(order_csv_path)
        else:
            mapping_df = pd.read_csv(order_csv_path, names=['Family', 'Order'])

        self.order_map = dict(zip(list(mapping_df['Family']), list(mapping_df['Order'])))
        self.master_df['order'] = self.master_df['family'].apply(lambda family_name: self.order_map[family_name])


    def load_gt(self, ground_truth_file, status_list):
        """
        Generates the ground truth from a scoring csv
        
        Arguments:
            ground_truth_file (str): filepath with ground truths
            status_list (List[str]): the list of status to load

        ground truth schema:    
            Filename: {catalog_number}.jpg
            {status} Status: TRUE/FALSE
            -- repeat for every status -- 

        Returns:
            ground_truth_df of all the desired ground truths.
                Schema:
                {status} Ground Truth: TRUE/FALSE
                -- repeat for every status --    

        """
        
        ground_truth_df = pd.read_csv(ground_truth_file)


        def substitute(value):
#             print(type(value))
#             print(value)
            if value == "True" or value == True:
                return False
            elif value == 'False' or value == False:
                return True
            elif np.isnan(value):
                return(np.nan)
            else:
                raise Exception("Invalid data type")


        for status in status_list:
                ground_truth_df.loc[:,f"{status} Status"] = ground_truth_df.loc[:,f"{status} Status"].apply(lambda x: substitute(x))

        ground_truth_df.rename(columns=dict(zip([f"{status} Status" for status in status_list],[f"{status} Ground Truth" for status in status_list])), inplace=True)
        
        ground_truth_df.loc[:, 'catalog_number'] = ground_truth_df['Filename'].apply(lambda filename: filename[:-4])
        ground_truth_df.set_index('catalog_number', inplace = True)
        gt_status_list = [f"{status} Ground Truth" for status in status_list]
        return ground_truth_df[gt_status_list]
    


    def load_preds(self, pred_csv_path, status_list, binarized = False):
        """
        Loads the predictions from pred_csv_path, binarizes the data, reindexes to catalog_number

        Arguments:
            pred_csv_path (str): path to the csv storing predictions
            status_list (List[str]): list of statuses to evaluate
            binarized (boolean): boolean of whether the data is binarized
        
        pred_csv_path schema:
            Filename: {catalog_number}.jpg
            {status} Status: TRUE/FALSE if binarized, {status}/Not_{status} if not binarized
            {status} status Confidence: float (>0.5)
            -- repeat for every status -- 

        Returns:
            pred_df  (pd.DataFrame): df with the predictions and confidence from status_list
                Schema:
                {status} Prediction: TRUE/FALSE
                {status} Prediction Confidence: float
                -- repeat for every status --  

        """
        pred_df = pd.read_csv(pred_csv_path)
        if np.isnan(pred_df.iloc[0,1]):
            pred_df = pred_df.loc[1:]

                        
        def substitute(value):
            if "Not" in value:
                return False
            else:
                return True

        if not binarized: 
            for status in status_list:
                pred_df.loc[:,f"{status} Prediction"] = pred_df.loc[:,f"{status} Status"].apply(lambda x: substitute(x))


        # extract catalog number and reindex to catalog number
        pred_df.loc[:, "Filename"] = pred_df['Filepath'].apply(lambda filepath: filepath.split('/')[-1])
        pred_df.loc[:, 'catalog_number'] = pred_df['Filename'].apply(lambda filename: filename[:-4])
        pred_df.set_index('catalog_number', inplace=True)

        pred_df.rename(columns=dict(zip([f"{status} Status Confidence" for status in status_list], [f"{status} Prediction Confidence" for status in status_list])), inplace=True)

        # only return desired statuses
        return_status_list = []
        for status in status_list:
            return_status_list.append(f"{status} Prediction")
            return_status_list.append(f"{status} Prediction Confidence")
        return pred_df[return_status_list]

   
      
    @staticmethod
    def parse_name(name) -> str:
#         if name[0] in {'1','2','3','4','5','6','7','8','9'}:
#             #print("entered starting with number")
#             if len(name) == 7:
#                 return "barcode-0"+name, name, int(name)
#             if len(name) == 6:
#                 return "barcode-00"+name, name, int(name)
        if name[0:4] == "ECON":
            partial = name[-7:]
            if partial[0] == '0':
                return "barcode-00"+name[-6:]
            else:
                return "barcode-0"+name[-7:]
#         if name[0:2] == "00":
#             print("barcode-"+name)
#             return "barcode-"+name
        if name[0:3] == "CBS":
            return "CBS." + name[3:9]
        if name[0:3] == "GH0":
            return "barcode-"+name[2:]
        if name[0:4] == "NEBC":
            partial = "barcode-0"+name[-7:]
            if partial[0] == "0":
                return "barcode-00"+partial[1:]
            else: 
                return partial
        if name[0:2] == "A0":
            partial = "barcode-0"+name[-7:]
            if partial[0] == "0":
                return "barcode-00"+partial[1:]
            else: 
                return partial
        if name[0:3] == "YU0":
            return "YU." + name[2:8]
        return name


    def merge_preds_gt(self, preds_df, gt_df):
        """
        Merges predictions and ground truth with self.master_df


        """
        merge_preds_gt_df = preds_df.join(gt_df, how='inner')
        merge_preds_gt_df['catalog_number'] = merge_preds_gt_df.index
        merge_preds_gt_df['catalog_number'] = merge_preds_gt_df['catalog_number'].apply(lambda x: Dataset.parse_name(x))
        merge_preds_gt_df.set_index('catalog_number', inplace=True)
        # display(merge_preds_gt_df)
        self.master_df.join(merge_preds_gt_df, how="left")
        # display(merge_preds_gt_df)
        # print(len(self.master_df))
        unmatched_preds = list(merge_preds_gt_df[~merge_preds_gt_df.index.isin(self.master_df.index)])
        # print(unmatched_preds[:min(len(unmatched_preds), 10)])
        # print(len(unmatched_preds))
        
        # retry unmatched_preds
        retry = [pred_catalog_number for pred_catalog_number in unmatched_preds if len(pred_catalog_number) == 8]
        # print(retry)
        merge_preds_gt_df['retry_catalog_number'] = merge_preds_gt_df[merge_preds_gt_df.index.isin(retry)].apply(lambda x: str(x[2:]))
        merge_preds_gt_df.set_index('retry_catalog_number', inplace=True)
        # display(merge_preds_gt_df.head(10))
        
        self.master_df.join(merge_preds_gt_df, how="inner")
        unmatched_preds = list(merge_preds_gt_df[~merge_preds_gt_df.index.isin(self.master_df.index)])
        # print(unmatched_preds[:min(len(unmatched_preds), 10)])
        # print(len(unmatched_preds))
