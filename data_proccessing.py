import pandas as pd
import numpy as np

def load_data():
    # Load the data
    data = pd.read_csv('data.csv')
    return data



rpm_all = pd.read_csv('tRNA_data_RPM_with_info.csv', index_col=0)

rpm_all = rpm_all.T

#seperate the data into two dataframes by column Place_taken: S or T
rpm_S = rpm_all[rpm_all['Place_taken'] == 'S']
rpm_T = rpm_all[rpm_all['Place_taken'] == 'T']

rpm_S = rpm_S.T
rpm_T = rpm_T.T

print(rpm_S.shape)
print(rpm_T.shape)

rpm_S.to_csv('S_RPM.csv')
rpm_T.to_csv('T_RPM.csv')



to_remove = []

# # Iterate over the rows of the DataFrame
# for g in cpm_all.index:
#     row_values = cpm_all.loc[g].astype(float)
#     percentile_85 = np.percentile(row_values, 85)
#     mean_value = row_values.mean()
#     median_value = row_values.median()

#     if median_value< 10:
#         to_remove.append(g)
    
#     # Append the index to to_remove if the condition is met
#     if percentile_85 < mean_value:
#         to_remove.append(g)

# # Filter out rows in to_remove from the DataFrame
# cpm_all = cpm_all.drop(to_remove)



