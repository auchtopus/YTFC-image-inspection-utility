import pandas as pd
import sys


if __name__ == "__main__":
    df = pd.read_csv(f"{sys.argv[1]}_dataset_1.csv")
    df['Filename'] = df['Filename'].apply(lambda x: f"{x}.jpg")
    df.to_csv(f"{sys.argv[1]}_dataset_1.csv")