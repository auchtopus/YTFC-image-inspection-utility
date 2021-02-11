import random

import pandas as pd
import numpy as np


def create_subsample(num):
    df = pd.read_csv('/Users/antonsquared/Projects/ytfc_image_utility/data/filter_master/dataset_2_full.csv')

    families = set(df['family'].dropna())

    sampling_dict = []
    sample_df = pd.DataFrame(columns = df.columns)
    for family in families:
        family_df = df[df['family']==family]
        sampling_dict.append({"family": family, "population": len(family_df), "sample": num})
        if len(family_df) < num:
            continue
            
        rand_select = random.sample(list(family_df.index.values), num)
        sample_df = pd.concat([sample_df, family_df.loc[rand_select]], ignore_index=True)

    comp_df = pd.DataFrame.from_dict(sampling_dict, orient='columns')
    for status in ["Reproductive", "Fruiting", "Flowering", "Budding"]:
        sample_df[f"{status} Ground Truth"] = np.nan
    sample_df.to_csv('02_09_2021_scoring.csv')
    comp_df.to_csv('02_09_2021_comp.csv')

create_subsample(5)

# upload later using aws cli