import json
from collections import OrderedDict
from pathlib import Path
import os

import psycopg2


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

    def __init__(self, session_key, local = True):
        if local:
            self.con = psycopg2.connect(host = "localhost", user = "antonsquared", password = "FuckPostgres1!", dbname = "image_access_utility")
        else:
            self.con = psycopg2.connect("postgres://ucqlxdrtocwgeo:8c5b38823520f7f6b643d8d481c199e3c46fcb39430d5f0778a18f8fa22a966d@ec2-3-231-194-96.compute-1.amazonaws.com:5432/d4cga65cton0kr")
        self.cur = self.con.cursor()
        self.cur.execute(f"""CREATE TABLE IF NOT EXISTS {session_key} (ID INT PRIMARY KEY, 
                                                                    REPRODUCTIVE BOOL, 
                                                                    FLOWERING BOOL, 
                                                                    FRUITING BOOL, 
                                                                    BUDDING BOOL)""")


        self.status_range = ["Reproductive", "Flowering", "Fruiting", "Budding"]
        self.session_key = session_key
        start_index = self.cur.execute(f"""SELECT max(ID) FROM {session_key}""")
        self.index = ss.get(val = start_index + 1) # go to the next index


        ## download the master file
        bucket.download_file(f'scoring_info/{session_key}/{session_key}_scoring.csv', f'{session_key}_scoring.csv')
        self.scoring_df = pd.read_csv(f'{session_key}_scoring.csv')
        self.length = len(self.scoring_df)
        
        # bucket.download_file(f'scoring_info/{session_key}_comp.csv', f'{session_key}_comp.csv')


    # replace this with a db!
    def submit(self, status_dict):
        old_id = self.cur.execute(f""" INSERT INTO {self.session_key} (ID, CATALOG_NUMBER, REPRODUCTIVE, FLOWERING, FRUITING, BUDDING) VALUES ({status_dict['index']},
                                                                                                                                               {status_dict['catalog_number']},
                                                                                                                                               {status_dict['Reproductive']},
                                                                                                                                               {status_dict['Flowering']},
                                                                                                                                               {status_dict['Fruiting']},
                                                                                                                                               {status_dict['Budding']})
        RETURNING ID""")   
        
        self.con.commit()

        self.index.val += 1

    def score(self):
        print(f"{self.index.val=}")

        if self.index.val == self.length:
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
    