import pandas as pd
import numpy as np

def load_data():
    # Load the data
    data = pd.read_csv('data.csv')
    return data

def data_mapping(data):
    # Mapping dictionary
    time_mapping = {'4h': 4,'24h': 24, '7d': 7 * 24 ,'0': 0}
    Sex_mapping = {'M': 0, 'F': 1}
    Treatment_mapping = {'CNT': 0, 'LPS': 1}
    Place_taken_mapping = {'S': 0, 'T': 1}

    # Convert the column using map
    data['Time_taken'] = data['Time_taken'].map(time_mapping)
    data['Sex'] = data['Sex'].map(Sex_mapping)
    data['Treatment'] = data['Treatment'].map(Treatment_mapping)
    data['Place_taken'] = data['Place_taken'].map(Place_taken_mapping)
    data['Sample_num'] = data['Sample_num'].str.replace('S', '')
    #data['Time_taken_normalized'] = data['Time_taken_normalized'].map(time_mapping)

    return data

# rpm_all = pd.read_csv('placetaken_S/S_RPM.csv', index_col=0)
# rpm_all = rpm_all.T

# data_mapping(rpm_all)

# rpm_all = rpm_all.T


# to_remove = []

# # Iterate over the rows of the DataFrame
# for g in rpm_all.index:
#     row_values = rpm_all.loc[g].astype(float)
#     percentile_85 = np.percentile(row_values, 85)
#     mean_value = row_values.mean()
#     median_value = row_values.median()

#     if median_value< 10:
#         to_remove.append(g)
    
#     # Append the index to to_remove if the condition is met
#     if percentile_85 < mean_value:
#         to_remove.append(g)

# # Filter out rows in to_remove from the DataFrame
# cpm_all = rpm_all.drop(to_remove)

# cpm_all.to_csv('placetaken_S/S_RPM_filtered.csv')

#samples_info = pd.read_csv('placetaken_T/T_Samples_info.csv', index_col=0)






