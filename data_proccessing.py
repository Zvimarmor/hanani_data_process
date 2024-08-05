import pandas as pd
import numpy as np

def load_data():
    # Load the data
    data = pd.read_csv('data.csv')
    return data



cpm_all = pd.read_csv('cpm_all_samples.csv', header=0, index_col=0)

to_remove = []

# Iterate over the rows of the DataFrame
for g in cpm_all.index:
    row_values = cpm_all.loc[g].astype(float)
    percentile_85 = np.percentile(row_values, 85)
    mean_value = row_values.mean()
    median_value = row_values.median()

    if median_value< 10:
        to_remove.append(g)
    
    # Append the index to to_remove if the condition is met
    if percentile_85 < mean_value:
        to_remove.append(g)

# Filter out rows in to_remove from the DataFrame
cpm_all = cpm_all.drop(to_remove)

cpm_all.to_csv('cpm_all_samples.csv')


