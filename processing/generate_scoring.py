import random

import pandas as pd
import numpy as np
df = pd.read_csv('/Users/antonsquared/Projects/ytfc_image_utility/data/filter_master/dataset_2_full.csv')

def create_subsample(num, df):
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
    # sample_df.to_csv('02_09_2021_scoring.csv')
    # comp_df.to_csv('02_09_2021_comp.csv')
    return sample_df, comp_df
df_new, df_comp = create_subsample(10, df)


def exclude(df_old, df_new):
    df_old.set_index(['cataldatog_number'], inplace = True)
    df_new.set_index(['catalog_number'], inplace=True)
    df_remains = df_new.drop(df_old.index.values, errors ='ignore')
    return df_remains

df_old = pd.read_csv('../02_09_2021_scoring.csv')

df_new = exclude(df_old, df_new)
df_new.reset_index(inplace=True)
df_final, df_comp_final = create_subsample(5, df_new)
df_final.to_csv('../dataset_2_02_24_2021_scoring.csv', index = True)
df_comp_final.to_csv('../dataset_2_02_24_2021_comp.csv')
