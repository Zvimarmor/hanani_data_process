import pandas as pd
import numpy as np

def load_data():
    # Load the data
    data = pd.read_csv('Samples_info.csv')
    return data


df = load_data()
#how to add a label to the columns? answer: df.columns = ['Sample', 'R1', 'R2', 'Ester', '001.flexbar', 'q.fastq']
df.columns = ['Sample', 'Time_taken', 'Treatment','Sex','Place_taken', 'Sample_id']

df.to_csv('Samples_info.csv', index=False)