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


import src.SessionState as ss

bucket_name = "ytfc"

s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)




class ScoringSession:

    def __init__(self, session_key):
        self.status_range = ["Reproductive", "Flowering", "Fruiting", "Budding"]
        self.session_key = session_key
        bucket.download_file(f"scoring_info/{session_key}/{session_key}.json", f"{session_key}.json")
        with open(f"{session_key}.json", 'r') as session_json:
            self.session = json.load(session_json)

        self.session_name = self.session['session_name']
        self.length = self.session['length']
        self.index = ss.get(val = self.session['index'])

        bucket.download_file(f'scoring_info/{session_key}/{session_key}_scoring.csv', f'{session_key}_scoring.csv')
        self.scoring_df = pd.read_csv(f'{session_key}_scoring.csv')
        
        # bucket.download_file(f'scoring_info/{session_key}_comp.csv', f'{session_key}_comp.csv')


    # replace this with a db!
    def submit(self, status_dict):
        self.session['index'] = status_dict['index']
        

        with open(f"{self.session_key}.json", 'w') as new_json:
            json.dump(self.session, new_json)
        bucket.upload_file(f"{self.session_key}.json", f"scoring_info/{self.session_key}/{self.session_key}.json")

        with open(f"{self.session_key}_{status_dict['catalog_number']}.json", 'w') as new_json:
            json.dump(status_dict, new_json)
        bucket.upload_file(f"{self.session_key}_{status_dict['catalog_number']}.json", f"scoring_info/{self.session_key}/{status_dict['catalog_number']}.json")

        self.index.val += 1

    def score(self):
        print(f"{self.index.val=}")

        if self.index.val == self.session['length'] + 1:
            st.write("Finished! Thank you :)")
            st.stop()

        col1, col2, col3, col4 = st.beta_columns(4)
        status_dict = dict(zip(self.status_range, [np.NaN] * 4))
        
        with col1:    
            status_dict['Reproductive'] = st.radio("Reproductive State:", [True, False, np.nan], index = 2)
        with col2:
            status_dict['Flowering']  = st.radio("Flowering State: ", [True, False, np.nan], index = 2)
        with col3:
            status_dict['Fruiting'] = st.radio("Fruiting State: ", [True, False, np.nan], index = 2)
        with col4:
            status_dict['Budding'] = st.radio("Budding State: ", [True, False, np.nan], index = 2)

        submit = st.button("Submit")

        if submit:
            status_save = status_dict.copy()
            status_save['catalog_number'] = self.scoring_df.loc[self.index.val, "catalog_number"]
            status_save['index'] = self.index.val
            self.submit(status_save)

        image_url = self.scoring_df.loc[self.index.val, "url"]
        image_name = self.scoring_df.loc[self.index.val, "catalog_number"]


        response = requests.get(image_url)
        sample_image = Image.open(BytesIO(response.content))
        st.image(sample_image, caption = f"sample: {image_name}", use_column_width=True) 
        st.text(f"{self.index.val}/{self.length} images scored")
    