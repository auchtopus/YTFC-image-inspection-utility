import pandas as pd
from src.views import Dataview


def summarize_dataset(status, path):
    dataset = Dataview(['Reproductive', 'Fruiting', 'Flowering', 'Budding'], {}, path)
    family_set = set(dataset.master_df['family'])
    full_dict = {}
    for family in family_set:
        summary_df, _ = dataset.summary_pd_query({"status": [status],
            "family": [family], "order": [None]}, ['Accuracy %', 'Capture %', 'Count'])
        full_dict[family] = {'Accuracy %': summary_df.loc[0.96, f'{status} Accuracy'], 'Capture %': summary_df.loc[0.96, f'{status} Capture'], 'Count': summary_df.loc[0.5, f'{status} Number of Samples']}

    summary_df, _ = dataset.summary_pd_query({"status": [status],
            "family": family_set, "order": [None]}, ['Accuracy %', 'Capture %', 'Count'])
    full_dict['Total'] =  {'Accuracy %': summary_df.loc[0.96, f'{status} Accuracy'], 'Capture %': summary_df.loc[0.96, f'{status} Capture'], 'Count': summary_df.loc[0.5, f'{status} Number of Samples']}
    result_df = pd.DataFrame.from_dict(full_dict, orient='index')

    print(result_df)


summarize_dataset("Reproductive", "/Users/antonsquared/Projects/ytfc_image_utility/data/filter_master/dataset_2.csv")