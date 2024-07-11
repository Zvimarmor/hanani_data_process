import pandas as pd
import numpy as np

def load_data():
    # Load the data
    data = pd.read_csv('Samples_info_numeric.csv', header=0, index_col=0)
    return data


df = load_data()
# Replace values in specific columns
df['Sex'].replace('M', 1, inplace=True)
df['Sex'].replace('F', 0, inplace=True)

df['Treatment'].replace('LPS', 1, inplace=True)
df['Treatment'].replace('CNT', 0, inplace=True)

df['Time_taken'].replace('24h', 24, inplace=True)
df['Time_taken'].replace('4h', 4, inplace=True)
df['Time_taken'].replace('7d', 168, inplace=True)

df.to_csv('Samples_info_numeric.csv', index=False)