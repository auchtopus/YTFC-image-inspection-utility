import os, sys, shutil
import json

import requests
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st
import boto3
from io import BytesIO

import src.SessionState as ss

bucket_name = "ytfc"


s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)

class ScoringInterface:

    def __init__(self):
        self.sessions = ["-", "02_09_2021"]


    def show(self):
        session_key = "02_09_2021"
        session = ScoringSession(session_key)
        session.score()



class ScoringSession:

    def __init__(self, session_key):
        bucket.download_file(f"scoring_info/{session_key}.json", f"{session_key}.json")
        with open(f"{session_key}.json", 'r') as session_json:
            self.session = json.load(session_json)

        self.session_name = self.session['session_name']
        self.length = self.session['length']
        self.index = ss.get(val = self.session['index'])

        bucket.download_file(f'scoring_info/{session_key}_scoring.csv', f'{session_key}_scoring.csv')
        self.scoring_df = pd.read_csv(f'{session_key}_scoring.csv')
        
        bucket.download_file(f'scoring_info/{session_key}_comp.csv', f'{session_key}_comp.csv')

    def score(self):
        st.button("hello!")
        if st.button("forward"):
            self.index.val +=1 

        if st.button("backward"):
            self.index.val -= 1


        image_url = self.scoring_df.loc[self.index.val, "url"]
        image_name = self.scoring_df.loc[self.index.val, "catalog_number"]



        response = requests.get(image_url)
        sample_image = Image.open(BytesIO(response.content))
        st.image(sample_image, caption = f"sample: {image_name}", use_column_width=True) 
        st.text(f"{self.index.val}/{self.length} images scored")

        

        # get image loading code from the other place!

        

    def save(self):
        save_button = st.button("Save")
        if st.button("Save"):
            # save scorings to s3, 
            pass



        