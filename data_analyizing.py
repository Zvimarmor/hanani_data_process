import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
from math import log10
import seaborn as sns
from scipy.stats import ttest_rel
from math import log2
from sklearn.model_selection import train_test_split

#######Data Preprocessing########

# # Read in the data
# ColData = pd.read_csv('Samples_info.csv', header=0, index_col=0)
# CountsDataFrame = pd.read_csv('tRNA_Exclusive_Combined_data.csv', header=0, index_col=0)

# #seperate the data into two file by place taken - S or T
# Samples_info_S = ColData[ColData['Place_taken'] == 'S']
# Samples_info_T = ColData[ColData['Place_taken'] == 'T']

# #save the data into csv files
# Samples_info_S.to_csv('Samples_info_S.csv')
# Samples_info_T.to_csv('Samples_info_T.csv')

#tool to keep only the wanted trfs
#trf_with_fdr_less_than_0_05 = ['tRF-28-P4R8YP9LOND5', 'tRF-28-86J8WPMN1E0J', 'tRF-30-RRJ89O9NF5W8', 'tRF-30-R9J89O9NF5W8', 
#                               'tRF-29-RRJ89O9NF5JP', 'tRF-28-PIR8YP9LOND5', 'tRF-30-86J8WPMN1E8Y', 'tRF-30-86V8WPMN1E8Y',
#                                 'tRF-28-86V8WPMN1E0J', 'tRF-29-P4R8YP9LONHK', 'tRF-29-86V8WPMN1EJ3', 'tRF-29-86J8WPMN1EJ3',
#                                   'tRF-31-XSXMSL73VL4YD', 'tRF-25-PS5P4PW3FJ', 'tRF-31-86J8WPMN1E8Y0', 'tRF-28-PER8YP9LOND5',
#                                     'tRF-17-W96KM8N', 'tRF-31-86V8WPMN1E8Y0', 'tRF-16-NS5J7KE', 'tRF-34-389MV47P596VJ5', 
#                                     'tRF-31-FSXMSL73VL4YD', 'tRF-30-PSQP4PW3FJI0', 'tRF-17-W9RKM8N']
#CountsDataFrame = CountsDataFrame[CountsDataFrame.index.isin(trf_with_fdr_les_than_0_05)]


# # Filter out rows where sum > 2
# CountsDataFrame = CountsDataFrame[CountsDataFrame.sum(axis=1) > 2]

# # Transpose the data frame for PCA
# Hanani_proccessed_data = CountsDataFrame.T

# # Add sample IDs as a column
# Hanani_proccessed_data['Sample_ID'] = Hanani_proccessed_data.index

# # Merge with sample information
# Hanani_proccessed_data = pd.merge(Hanani_proccessed_data, ColData, left_on='Sample_ID', right_index=True)

# # Set index to Sample_ID
# Hanani_proccessed_data.set_index('Sample_ID', inplace=True)

# # Remove non-numeric columns for PCA
# non_numeric_columns = ['Time_taken', 'Treatment', 'Sex', 'Place_taken', 'Sample_num']
# numeric_columns = [col for col in Hanani_proccessed_data.columns if col not in non_numeric_columns]

# cpm_all = pd.read_csv('cpm_all_samples.csv', header=0, index_col=0)
#Hanani_proccessed_data_S = pd.read_csv('placetaken_S/S_RPM.csv', header=0, index_col=0)

#######Plotting########

def plot_with_legend(data, x, y, labels, title, xlabel, ylabel, label_mapping):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data[:, x], data[:, y], c=labels, cmap='viridis')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # Create a legend
    unique_labels = np.unique(labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label_mapping[label], 
                          markersize=10, markerfacecolor=plt.cm.viridis(i / max(unique_labels))) 
               for i, label in enumerate(unique_labels)]
    plt.legend(handles = handles, title='Treatment & Time Taken', loc='best')
    plt.show()
    plt.close()



#######PCA, TSNE, LDA########

# # Perform PCA
# pca = PCA(n_components=4, whiten=True)
# pca_transformed = pca.fit_transform(Hanani_proccessed_data[numeric_columns])

# # Perform t-SNE
# tsne = TSNE(n_components=2, perplexity=30, random_state=0)
# tsne_transformed = tsne.fit_transform(Hanani_proccessed_data[numeric_columns])

# # Create a combined label for Treatment, Sex, and Place_taken
# Hanani_proccessed_data['Combined_Label'] = Hanani_proccessed_data['Treatment'].astype(str) + '&' + Hanani_proccessed_data['Time_taken'].astype(str)

# # Encode combined labels
# combined_label_encoder = LabelEncoder()
# Hanani_proccessed_data['Combined_Label'] = combined_label_encoder.fit_transform(Hanani_proccessed_data['Combined_Label'])

# # perform LDA on the data
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# lda = LinearDiscriminantAnalysis(n_components=2)
# lda_transformed = lda.fit_transform(Hanani_proccessed_data[numeric_columns], Hanani_proccessed_data['Combined_Label'])

# # Get the mapping from encoded labels to original combined labels
# label_mapping = dict(zip(Hanani_proccessed_data['Combined_Label'], Hanani_proccessed_data['Combined_Label'].astype(str) ))

# # Plot first vs. second principal component from PCA
# plot_with_legend(pca_transformed, 0, 1, Hanani_proccessed_data['Combined_Label'], 'PCA: PC1 vs PC2', 'PC1', 'PC2', label_mapping)

# # Plot third vs. fourth principal component from PCA
# plot_with_legend(pca_transformed, 2, 3, Hanani_proccessed_data['Combined_Label'], 'PCA: PC3 vs PC4', 'PC3', 'PC4', label_mapping)

# # Plot t-SNE results
# plot_with_legend(tsne_transformed, 0, 1, Hanani_proccessed_data['Combined_Label'], 't-SNE', 't-SNE 1', 't-SNE 2', label_mapping)

# # Plot LDA results
# plot_with_legend(lda_transformed, 0, 1, Hanani_proccessed_data['Combined_Label'], 'LDA', 'LDA 1', 'LDA 2', label_mapping)


#######clustering methods########
# def plot_cluster(x, y, title, xlabel, ylabel, labels, data=Hanani_proccessed_data, jitter=True):
#     # Ensure the DataFrame is being used properly
#     if jitter:
#         jitter_strength = 0.1
#         jittered_x = data.iloc[:, x] + np.random.uniform(-jitter_strength, jitter_strength, data.shape[0])
#         jittered_y = data.iloc[:, y] + np.random.uniform(-jitter_strength, jitter_strength, data.shape[0])
#     else:
#         jittered_x = data.iloc[:, x]
#         jittered_y = data.iloc[:, y]

#     # Plotting
#     plt.figure(figsize=(18, 8))
#     plt.scatter(jittered_x, jittered_y, c=labels, cmap='viridis', s=100, alpha=0.6)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.show()

# def print_cluster_info(data, labels, cluster_column):
#     unique_clusters = np.unique(labels)
#     for cluster in unique_clusters:
#         print('Cluster: ', cluster)
#         print('num of samples: ', data[data[cluster_column] == cluster].shape[0])
#         print('LPS: ', data[(data[cluster_column] == cluster) & (data['Treatment'] == 1)].shape[0])
#         print('ctrl: ', data[(data[cluster_column] == cluster) & (data['Treatment'] == 0)].shape[0])
#         print('4h: ', data[(data[cluster_column] == cluster) & (data['Time_taken'] == 4)].shape[0])
#         print('24h: ', data[(data[cluster_column] == cluster) & (data['Time_taken'] == 24)].shape[0])
#         print('7d: ', data[(data[cluster_column] == cluster) & (data['Time_taken'] == 7 * 24)].shape[0])

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
    data['Time_taken_normalized'] = data['Time_taken_normalized'].map(time_mapping)

    return data

# #mapping the data
# Hanani_proccessed_data = data_mapping(Hanani_proccessed_data)



#######K means clustering########
# from sklearn.cluster import KMeans

# # Apply K-means clustering
# kmeans = KMeans(n_clusters=3, random_state=0)
# kmeans.fit(Hanani_proccessed_data)

# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
# Hanani_proccessed_data['kmeans_cluster'] = labels

# # Plotting K-means clustering results
# plot_cluster(0, 1, 'K-means Clustering', 'PC1', 'PC2', labels=Hanani_proccessed_data['kmeans_cluster'])
# print_cluster_info(Hanani_proccessed_data, labels, 'kmeans_cluster')



# #######Spectral Clustering########
# from sklearn.cluster import SpectralClustering

# # Initialize SpectralClustering
# spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0)
# spectral_labels = spectral.fit_predict(Hanani_proccessed_data[numeric_columns])

# # Add SpectralClustering labels to the data
# Hanani_proccessed_data['Spectral_Cluster'] = spectral_labels

# # Plotting Spectral Clustering results
# plot_cluster(0, 1, 'Spectral Clustering', 'Feature 1', 'Feature 2', labels=Hanani_proccessed_data['Spectral_Cluster'])
# print_cluster_info(Hanani_proccessed_data, Hanani_proccessed_data['Spectral_Cluster'], 'Spectral_Cluster')



# #######Agglomerative Clustering########
# from sklearn.cluster import AgglomerativeClustering

# # Initialize AgglomerativeClustering
# agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')

# # Fit the model
# agg_labels = agg_clustering.fit_predict(Hanani_proccessed_data[numeric_columns])

# # Add AgglomerativeClustering labels to the data
# Hanani_proccessed_data['Agglomerative_Cluster'] = agg_labels

# # Plotting Agglomerative Clustering results
# plot_cluster(0, 1, 'Agglomerative Clustering', 'Feature 1', 'Feature 2', labels=Hanani_proccessed_data['Agglomerative_Cluster'])
# print_cluster_info(Hanani_proccessed_data, Hanani_proccessed_data['Agglomerative_Cluster'], 'Agglomerative_Cluster')

# # #######Hierarchical Clustering########
# from scipy.cluster.hierarchy import linkage, fcluster
# import scipy.cluster.hierarchy as sch

# # Compute the linkage matrix
# Z = linkage(Hanani_proccessed_data[numeric_columns], method='ward')

# # Form clusters
# hierarchical_labels = fcluster(Z, t=3, criterion='maxclust')

# # Add Hierarchical labels to the data
# Hanani_proccessed_data['Hierarchical_Cluster'] = hierarchical_labels

# # Plotting Hierarchical Clustering results
# plot_cluster(0, 1, 'Hierarchical Clustering', 'Feature 1', 'Feature 2', labels=Hanani_proccessed_data['Hierarchical_Cluster'])
# print_cluster_info(Hanani_proccessed_data, Hanani_proccessed_data['Hierarchical_Cluster'], 'Hierarchical_Cluster')


#######SVM Classification########
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split

# X, y = Hanani_proccessed_data.drop('Treatment', axis=1), Hanani_proccessed_data['Treatment']

# test_sizes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,0.45,0.5]
# train_sizes = [1 - test_size for test_size in test_sizes]
# accuracies = []

# for test_size in test_sizes:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=52)
#     model = SVC(kernel='linear')
#     model.fit(X_train, y_train)
#     accuracies.append(model.score(X_test, y_test))
#     print('test size:', test_size,'accuracy:', model.score(X_test, y_test), 'num of right predictions: ', model.score(X_test, y_test) * X_test.shape[0],'out of:', X_test.shape[0])

# plt.plot(train_sizes, accuracies)
# plt.xlabel('Train Size (out of 1)')
# plt.ylabel('Accuracy') 
# plt.title('SVM Classification Accuracy vs. Test Size')
# plt.show()
# plt.close()

#######Random Forest Classification########
# from sklearn.ensemble import RandomForestClassifier

# accuracies = []

# for test_size in test_sizes:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
#     model = RandomForestClassifier(n_estimators=10)
#     model.fit(X_train, y_train)
#     accuracies.append(model.score(X_test, y_test))
#     print('test size:', test_size,'accuracy:', model.score(X_test, y_test), 'num of right predictions: ', model.score(X_test, y_test) * X_test.shape[0],'out of:', X_test.shape[0])

#######perceotron Classification########
# from sklearn.linear_model import Perceptron

# accuracies = []

# for test_size in test_sizes:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
#     model = Perceptron()
#     model.fit(X_train, y_train)
#     accuracies.append(model.score(X_test, y_test))
#     print('test size:', test_size,'accuracy:', model.score(X_test, y_test), 'num of right predictions: ', model.score(X_test, y_test) * X_test.shape[0],'out of:', X_test.shape[0])


#######knn Classification########
# from sklearn.neighbors import KNeighborsClassifier

# accuracies = []
# k_values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# for value in k_values:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
#     model = KNeighborsClassifier(n_neighbors=value)
#     model.fit(X_train, y_train)
#     accuracies.append(model.score(X_test, y_test))
    #print('num of neighbors:', value,'accuracy:', model.score(X_test, y_test), 'num of right predictions: ', model.score(X_test, y_test) * X_test.shape[0],'out of:', X_test.shape[0])

# plt.plot(k_values, accuracies)
# plt.xlabel('Number of Neighbors')
# plt.ylabel('Accuracy')
# plt.title('KNN Classification Accuracy vs. Number of Neighbors')
# plt.show()
# plt.close()

#######Gradient Boosting Classification########
# from sklearn.ensemble import GradientBoostingClassifier

# accuracies = []

# for test_size in test_sizes:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
#     model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
#     model.fit(X_train, y_train)
#     accuracies.append(model.score(X_test, y_test))
#     print('test size:', test_size,'accuracy:', model.score(X_test, y_test), 'num of right predictions: ', model.score(X_test, y_test) * X_test.shape[0],'out of:', X_test.shape[0])

# plt.plot(train_sizes, accuracies)
# plt.xlabel('Train Size (out of 1)')
# plt.ylabel('Accuracy')
# plt.title('Gradient Boosting Classification Accuracy vs. Test Size')
# plt.show()
# plt.close()


#######rpm data analysis per each Ganglion########

# rpm = pd.read_csv('placetaken_T/T_RPM.csv', header=0, index_col=0)

# Time_taken_normalized = [0,0,0,0,24,24,24,24,0,0,0,0,4,4,4,4,0,0,0,0,168,168,168,168]
# not_to_include = ['Time_taken', 'Treatment','Sex','Place_taken','Sample_num', 'Time_taken_normalized']

# # Initialize a list to store the correlations
# correlations = []
# p_values = []

# # Calculate pearson correlation for each row with the 'Time_taken_normalized' column
# for index, row in rpm.iterrows():
#     if index not in not_to_include:
#         numeric_values = row.astype(float).values
#         correlation, p_value = pearsonr(numeric_values, Time_taken_normalized)
#         correlations.append(correlation)
#         p_values.append(p_value)
#     else:
#         # For excluded rows, append NaN or some placeholder value
#         correlations.append(float('nan'))
#         p_values.append(float('nan'))

# # Add the correlations as a new column in the original DataFrame
# rpm['Pearson_correlation'] = correlations
# rpm['p_values'] = p_values

# rpm.to_csv('placetaken_T/T_RPM_with_p_val&pearson.csv')

# all_rpm = pd.read_csv('placetaken_T/T_RPM_with_p_val&pearson.csv', header=0, index_col=0)

# # Read the entire CSV file into a single column
# df = pd.read_csv('tRF_meta.csv', usecols=[0, 1, 2, 3,4], header=0, index_col=0)

# print(df.head())


# # Now, merge with all_rpm DataFrame and merge the columns head from the two DataFrames
# merged = pd.merge(all_rpm, df, on='Trfs', how='inner')

# # Display the merged DataFrame
# print(merged.head())

# # Save the merged DataFrame to a new CSV file
# merged.to_csv('placetaken_T/T_RPM_with_pearson_and_meta.csv', index=True)

# # Read the CSV file into a DataFrame
# df = pd.read_csv('placetaken_T/T_RPM_with_pearson_and_meta.csv')

# # Extract the 'P_value' column for FDR correction
# p_values = df['p_values'].values

# # Perform FDR correction using the Benjamini-Hochberg method
# _, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')


# # Add the FDR-adjusted p-values back to the DataFrame as a new column after the original p-values column (column number 51)
# df.insert(27, 'FDR_corrected_p_values', pvals_corrected)

# # Save the updated DataFrame to a new CSV file and the trfs names as first column
# df.to_csv('placetaken_T/T_RPM_with_correlation_and_meta.csv', index=False)

##plotting the correlation by the p-values
# p_values = df['p_values'].values
# for i in range(len(p_values)):
#     p_values[i] = -log10(p_values[i])
# correlations = df['Spearman_correlation'].values
# trf_types = df['tRF_type(s)']

# # Automatically assign colors to each unique tRF type
# unique_trf_types = trf_types.unique()
# colors = sns.color_palette('hsv', len(unique_trf_types))
# color_mapping = dict(zip(unique_trf_types, colors))
# color_assigned = trf_types.map(color_mapping)

# plt.figure(figsize=(12,8))
# plt.scatter(correlations, p_values, c=color_assigned, alpha=0.5)

# # Create legend handles
# handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[trf], markersize=8) 
#            for trf in unique_trf_types]
# plt.legend(handles, unique_trf_types, title="tRF Type", bbox_to_anchor=(1.05, 1), loc='best')

# plt.xlabel('Spearman Correlation')
# plt.ylabel('-log10(P-value)')
# plt.title('Spearman Correlation vs. -log10(P-value)')
# plt.show()
# plt.close()

##plotting the rpm vs time taken##

# df1 = pd.read_csv('placetaken_S/S_RPM_pearson_FDR_meta.csv')
# df2 = pd.read_csv('placetaken_T/T_RPM_pearson_FDR_meta.csv')

# # Correct the filtering syntax
# significant_trfs1 = df1[(df1['p_values'] < 0.05) & (np.abs(df1['Pearson_correlation']) > 0.5)]
# significant_trfs2 = df2[(df2['p_values'] < 0.05) & (np.abs(df2['Pearson_correlation']) > 0.5)]

# print('T ganglion')
# print(significant_trfs1)
# print(len(significant_trfs1))
# print('************************')
# print('S ganglion')
# print(significant_trfs2)
# print(len(significant_trfs2))

# # Save the significant tRFs to a new CSV file 
# significant_trfs1.to_csv('significant_trfs_T_pearson.csv', index=False)
# significant_trfs2.to_csv('significant_trfs_S_pearson.csv', index=False)


# df = df.set_index('Trfs')

# yaxis= [-0.4063840514725541,1.083995211456004,-1.184449696089669,0.5068385361062193,-0.7544238213475143,1.3558300031970427,0.15301763949798586,-0.7544238213475143,-0.7608150739635381,1.3458969882795695,-0.7608150739635381,0.1757331596475068,-0.7858269900332325,1.4584125360878775,-0.21520776582943774,-0.4573777802252062,-0.38553089160438053,1.495006945409672,-0.575373043402434,-0.5341030104028572,0.7738206322717607,-1.4471880868877005,0.1365132927117562,0.536854161904184]
# Time_taken = [24,24,24,24,24,24,24,24,4,4,4,4,4,4,4,4,168,168,168,168,168,168,168,168]
# Treatment = ['CNT','CNT','CNT','CNT','LPS','LPS','LPS','LPS','CNT','CNT','CNT','CNT','LPS','LPS','LPS','LPS','CNT','CNT','CNT','CNT','LPS','LPS','LPS','LPS']
# interaction = ['CNT_24', 'CNT_24', 'CNT_24', 'CNT_24', 'LPS_24', 'LPS_24', 'LPS_24', 'LPS_24', 'CNT_4', 'CNT_4', 'CNT_4', 'CNT_4', 'LPS_4', 'LPS_4', 'LPS_4', 'LPS_4', 'CNT_168', 'CNT_168', 'CNT_168', 'CNT_168', 'LPS_168', 'LPS_168', 'LPS_168', 'LPS_168']
# colors= ['blue','blue','blue','blue','red','red','red','red','blue','blue','blue','blue','red','red','red','red','blue','blue','blue','blue','red','red','red','red']

# #create a list of tuples with the data
# data = list(zip(yaxis, interaction))

# sort_by = ['CNT_4','LPS_4','CNT_24','LPS_24','CNT_168','LPS_168']
# data.sort(key=lambda x: sort_by.index(x[1]))

# #extract the data into two lists
# yaxis, interaction = zip(*data)

# plt.scatter(interaction, yaxis, c=colors)
# plt.xlabel('tRF-28-MIF91SS2P4DX')
# plt.ylabel('Time_taken_normalized and Treatment')
# plt.title('tRF-28-MIF91SS2P4DX vs. Time_taken_normalized and Treatment in the T ganglion after zscore normalization')
# plt.show()
# plt.close()

##doing z score normalization (by hand)###

# s_rpm = pd.read_csv('placetaken_S/S_RPM.csv', header=0, index_col=0)
# normalized_rpm = pd.DataFrame()

# F_24 = [0,1,4,7]
# M_24 = [2,3,5,6]
# F_4 = [8,9,12,13]
# M_4 = [10,11,14,15]
# F_168 = [16,17,20,21]
# M_168 = [18,19,22,23]

# for index, row in s_rpm.iterrows():
#     #calculate for F_24
#     mean = row.iloc[F_24].mean()
#     std = row.iloc[F_24].std()
#     for i in F_24:
#         normalized_rpm.loc[index, i] = (row.iloc[i] - mean) / std
    
#     # Calculate for M_24
#     mean = row.iloc[M_24].mean()
#     std = row.iloc[M_24].std()
#     for i in M_24:
#         normalized_rpm.loc[index, i] = (row.iloc[i] - mean) / std

#     # Calculate for F_4
#     mean = row.iloc[F_4].mean()
#     std = row.iloc[F_4].std()
#     for i in F_4:
#         normalized_rpm.loc[index, i] = (row.iloc[i] - mean) / std

#     # Calculate for M_4
#     mean = row.iloc[M_4].mean()
#     std = row.iloc[M_4].std()
#     for i in M_4:
#         normalized_rpm.loc[index, i] = (row.iloc[i] - mean) / std

#     # Calculate for F_168
#     mean = row.iloc[F_168].mean()
#     std = row.iloc[F_168].std()
#     for i in F_168:
#         normalized_rpm.loc[index, i] = (row.iloc[i] - mean) / std

#     # Calculate for M_168
#     mean = row.iloc[M_168].mean()
#     std = row.iloc[M_168].std()
#     for i in M_168:
#         normalized_rpm.loc[index, i] = (row.iloc[i] - mean) / std

# normalized_rpm.to_csv('placetaken_S/S_RPM_zscore.csv')

###doing t test for the z score normalized data###

# s_rpm = pd.read_csv('placetaken_T/T_RPM.csv', header=0, index_col=0)

# # Initialize a list to store the p-values
# p_values_168 = []
# significant_trfs = []
# log2_fold_changes = []

# # Perform t-test for each row
# for index, row in s_rpm.iterrows():
#     #LPS = row[[4,5,6,7,12,13,14,15,20,21,22,23]]
#     #CNT = row[[0,1,2,3,8,9,10,11,16,17,18,19]]
#     LPS = row[[20,21,22,23]]
#     CNT = row[[16,17,18,19]]
#     t_statistic, p_value = ttest_rel(LPS, CNT)
#     LPS_mean = LPS.mean()
#     CNT_mean = CNT.mean()
#     log2_fold_change = log2((LPS_mean+1) / (CNT_mean+1))
#     log2_fold_changes.append(log2_fold_change)
#     p_values_168.append(p_value)

# # Perform FDR correction using the Benjamini-Hochberg method
# _, pvals_corrected, _, _ = multipletests(p_values_168, method='fdr_bh')

# for i in range(len(p_values_168)):
#     if p_values_168[i] < 0.05:
#         significant_trfs.append(s_rpm.index[i])

# s_rpm['p_values'] = p_values_168
# s_rpm['FDR_corrected_p_values'] = pvals_corrected
# s_rpm['log2_fold_changes'] = log2_fold_changes

# print(significant_trfs)
# print(len(significant_trfs))
# df = pd.read_csv('tRF_meta.csv', usecols=[0, 1, 2, 3,4], header=0, index_col=0)

# significant_trfs_with_meta = s_rpm[s_rpm.index.isin(significant_trfs)]
# significant_trfs_with_meta = pd.merge(significant_trfs_with_meta, df, left_index=True, right_index=True)
# significant_trfs_with_meta.to_csv('placetaken_T/T_RPM_with_p_val&meta.csv')


#s_rpm.to_csv('placetaken_S/S_RPM_with_p_val.csv')


# #plotting the rpm vs time taken in the T ganglion
# t_rpm = pd.read_csv('placetaken_T/T_RPM_with_p_val&meta.csv', header=0, index_col=0)

# # Extract the 'log2_fold_changes' column for FDR correction
# log2_fold_changes_t = t_rpm['log2_fold_changes'].values

# p_values_t = t_rpm['p_values'].values

# p_values_t = [-log10(p) for p in p_values_t]

# # Automatically assign colors to each unique tRF type
# trf_types = t_rpm['locations']
# unique_trf_types = trf_types.unique()
# colors = sns.color_palette('hsv', len(unique_trf_types))
# color_mapping = dict(zip(unique_trf_types, colors))
# color_assigned = trf_types.map(color_mapping)

# plt.figure(figsize=(12,8))
# plt.scatter(log2_fold_changes_t, p_values_t, c=color_assigned, alpha=0.5)

# handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[trf], markersize=8) 
#            for trf in unique_trf_types]
# plt.legend(handles, unique_trf_types, title="tRF Type", bbox_to_anchor=(1.05, 1), loc='best')

# plt.xlabel('log2_fold_changes')
# plt.ylabel('-log10(P-value)')
# plt.title('log2_fold_changes vs. -log10(P-value) in the T ganglion')
# plt.savefig('log2_fold_changes_vs_-log10(P-value)_T_ganglion.png')
# plt.show()
# plt.close()

# #plotting the rpm vs time taken in the S ganglion

# s_rpm = pd.read_csv('placetaken_S/S_RPM_with_p_val&meta.csv', header=0, index_col=0)

# log2_fold_changes_s = s_rpm['log2_fold_changes'].values

# p_values_s = s_rpm['p_values'].values  

# p_values_s = [-log10(p) for p in p_values_s]

# trf_types = s_rpm['locations']
# unique_trf_types = trf_types.unique()
# colors = sns.color_palette('hsv', len(unique_trf_types))
# color_mapping = dict(zip(unique_trf_types, colors))
# color_assigned = trf_types.map(color_mapping)

# plt.figure(figsize=(12,8))
# plt.scatter(log2_fold_changes_s, p_values_s, c=color_assigned, alpha=0.5)

# handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[trf], markersize=8)
#               for trf in unique_trf_types]
# plt.legend(handles, unique_trf_types, title="tRF Type", bbox_to_anchor=(1.05, 1), loc='best')

# plt.xlabel('log2_fold_changes')
# plt.ylabel('-log10(P-value)')
# plt.title('log2_fold_changes vs. -log10(P-value) in the S ganglion')
# plt.savefig('log2_fold_changes_vs_-log10(P-value)_S_ganglion.png')
# plt.show()
# plt.close()

##checking the percentage of each protein in the S and T ganglion##

# s_edger = pd.read_csv('placetaken_S/EdgeR_S_ganglion.csv', header=0, index_col=0)
# t_edger = pd.read_csv('placetaken_T/EdgeR_T_ganglion.csv', header=0, index_col=0)

# protein_dict_s = dict()
# protein_dict_t = dict()
# for index, row in s_edger.iterrows():
#     if row['trna'] not in protein_dict_s:
#         protein_dict_s[row['trna']] = 1
#     else:
#         protein_dict_s[row['trna']] += 1

# for index, row in t_edger.iterrows():
#     if row['trna'] not in protein_dict_t:
#         protein_dict_t[row['trna']] = 1
#     else:
#         protein_dict_t[row['trna']] += 1

# s_len = s_edger.shape[0]
# t_len = t_edger.shape[0]

# print('S ganglion length:', s_len)
# print('S ganglion:')
# for protein in protein_dict_s:
#     print('protein:', protein, 'num of times:', protein_dict_s[protein], 'percentage:', (protein_dict_s[protein] / (s_len)) * 100)

# print('*************************************')

# print('T ganglion length:', t_len)
# print('T ganglion:')
# for protein in protein_dict_t:
#     print('protein:', protein, 'num of times:', protein_dict_t[protein], 'percentage:', (protein_dict_t[protein] / (t_len)) * 100)

conclusion_txt_file = open('conclusion.txt', 'w')

s_4_significant = pd.read_csv('placetaken_S/sg_S4_Genes (1).csv', header=0, index_col=0)
s_24_significant = pd.read_csv('placetaken_S/sg_S24_Genes (1).csv', header=0, index_col=0)
s_168_significant = pd.read_csv('placetaken_S/sg_S168_Genes (1).csv', header=0, index_col=0)
t_4_significant = pd.read_csv('placetaken_T/sg_T4_Genes (1).csv', header=0, index_col=0)
t_24_significant = pd.read_csv('placetaken_T/sg_T24_Genes (1).csv', header=0, index_col=0)
t_168_significant = pd.read_csv('placetaken_T/sg_T168_Genes (1).csv', header=0, index_col=0)

s_4_significant = s_4_significant[s_4_significant['PValue'] < 0.05]
s_24_significant = s_24_significant[s_24_significant['PValue'] < 0.05]
s_168_significant = s_168_significant[s_168_significant['PValue'] < 0.05]
t_4_significant = t_4_significant[t_4_significant['PValue'] < 0.05]
t_24_significant = t_24_significant[t_24_significant['PValue'] < 0.05]
t_168_significant = t_168_significant[t_168_significant['PValue'] < 0.05]

s_4_significant = s_4_significant.drop(columns=["LR","log2FC","seq","details","len","codon","gene","origin","isSg"])
s_24_significant = s_24_significant.drop(columns=["LR","log2FC","seq","details","len","codon","gene","origin","isSg"])
s_168_significant = s_168_significant.drop(columns=["LR","log2FC","seq","details","len","codon","gene","origin","isSg"])
t_4_significant = t_4_significant.drop(columns=["LR","log2FC","seq","details","len","codon","gene","origin","isSg"])
t_24_significant = t_24_significant.drop(columns=["LR","log2FC","seq","details","len","codon","gene","origin","isSg"])
t_168_significant = t_168_significant.drop(columns=["LR","log2FC","seq","details","len","codon","gene","origin","isSg"])


conclusion_txt_file.write('S Ganglion:' + '\n'+ '\n')
conclusion_txt_file.write('S ganglion 4h significant:'+'\n' + str(s_4_significant) + '\n'+'len:' + str(len(s_4_significant)) + '\n' + '\n')
conclusion_txt_file.write('S ganglion 24h significant:'+'\n' + str(s_24_significant) + '\n'+'len:' + str(len(s_24_significant)) + '\n'+ '\n')
conclusion_txt_file.write('S ganglion 168h significant:'+'\n' + str(s_168_significant) + '\n'+'len:' + str(len(s_168_significant)) + '\n'+ '\n')

conclusion_txt_file.write('T Ganglion:' + '\n'+ '\n')
conclusion_txt_file.write('T ganglion 4h significant:'+'\n' + str(t_4_significant) + '\n'+'len:' + str(len(t_4_significant)) + '\n'+ '\n')
conclusion_txt_file.write('T ganglion 24h significant:'+'\n' + str(t_24_significant) + '\n'+'len:' + str(len(t_24_significant)) + '\n'+ '\n')
conclusion_txt_file.write('T ganglion 168h significant:'+'\n' + str(t_168_significant) + '\n'+'len:' + str(len(t_168_significant)) + '\n'+ '\n')


conclusion_txt_file.write('*'*20 + '\n')

Tganglion_4_24_significant = []
Tganglion_alltimes_significant = []
for trf in s_4_significant['trf']:
    if trf in t_24_significant['trf'].values and trf in t_168_significant['trf'].values:
        Tganglion_alltimes_significant.append(trf)
    if trf in t_24_significant['trf'].values:
        Tganglion_4_24_significant.append(trf)

conclusion_txt_file.write('T ganglion 4 and 24 significant:' + str(Tganglion_4_24_significant) + '\n')
conclusion_txt_file.write('T ganglion all times significant:' + str(Tganglion_alltimes_significant)+ '\n'+ '\n')

Sganglion_alltimes_significant = []
Sganglion_4_24_significant = []
for trf in s_4_significant['trf']:
    if trf in s_24_significant['trf'].values and trf in s_168_significant['trf'].values:
        Sganglion_alltimes_significant.append(trf)
    if trf in s_24_significant['trf'].values:
        Sganglion_4_24_significant.append(trf)


conclusion_txt_file.write('S ganglion 4 and 24 significant:' + str(Sganglion_4_24_significant) + '\n')
conclusion_txt_file.write('S ganglion all times significant:' + str(Sganglion_alltimes_significant) +'\n')
conclusion_txt_file.write('*'*20 + '\n')

for trf in Tganglion_4_24_significant:
    if trf in Sganglion_4_24_significant:
        conclusion_txt_file.write('tRFs significant in both S and T ganglion in 4 and 24 hours:' + str(trf) + '\n')

for trf in Tganglion_alltimes_significant:
    if trf in Sganglion_alltimes_significant:
        conclusion_txt_file.write('tRFs significant in both S and T ganglion in all times:' + str(trf) + '\n')

conclusion_txt_file.close()

## plotting a ven diagram for the significant trfs in the T ganglion##

from matplotlib_venn import venn2, venn3

s_4_significant = s_4_significant[s_4_significant['PValue'] < 0.05]
s_24_significant = s_24_significant[s_24_significant['PValue'] < 0.05]
s_168_significant = s_168_significant[s_168_significant['PValue'] < 0.05]
t_4_significant = t_4_significant[t_4_significant['PValue'] < 0.05]
t_24_significant = t_24_significant[t_24_significant['PValue'] < 0.05]
t_168_significant = t_168_significant[t_168_significant['PValue'] < 0.05]

all_significant_data_s = pd.concat([s_4_significant, s_24_significant, s_168_significant])
all_significant_data_t = pd.concat([t_4_significant, t_24_significant, t_168_significant])

s_4_significant = s_4_significant['trf']
s_24_significant = s_24_significant['trf']
s_168_significant = s_168_significant['trf']
t_4_significant = t_4_significant['trf']
t_24_significant = t_24_significant['trf']
t_168_significant = t_168_significant['trf']

s_4_significant = set(s_4_significant)
s_24_significant = set(s_24_significant)
s_168_significant = set(s_168_significant)
t_4_significant = set(t_4_significant)
t_24_significant = set(t_24_significant)
t_168_significant = set(t_168_significant)

all_s = s_4_significant | s_24_significant | s_168_significant
all_t = t_4_significant | t_24_significant | t_168_significant

christof_data = pd.read_csv('Blood_tRF_christof.csv', header=0, index_col=0)
christof_data = christof_data[christof_data['pvalue'] < 0.05]
#christof_data = christof_data['trf']
christof_data_set = set(christof_data['trf'])

plt.figure(figsize=(30, 30))
#venn = venn3([s_4_significant, s_24_significant, s_168_significant], set_labels=('4h', '24h', '168h'))
#venn = venn3([s_4_significant, t_24_significant, t_168_significant], set_labels=('4h', '24h', '168h'))
venn = venn3([all_s, all_t, christof_data_set], set_labels=('S ganglion', 'T ganglion', 'Christof data'))


common_trfs = all_s & all_t & christof_data_set
sandt = all_s & all_t
sandc = all_s & christof_data_set
tandc = all_t & christof_data_set
labels_for_trfs = dict()

# Create conditions for coloring
common_trf_labels = ''
for trf in common_trfs:
    # Find the corresponding rows for the TRF in both sets
    s_value = all_significant_data_s.loc[all_significant_data_s['trf'] == trf, 'logFC'].values[0]  # LogFC value from Set S
    t_value = all_significant_data_t.loc[all_significant_data_t['trf'] == trf, 'logFC'].values[0]  # LogFC value from Set T
    christof_value = christof_data.loc[christof_data['trf'] == trf, 'log2FoldChange'].values[0]  # LogFC value from Christof data

    # Check the conditions for coloring

    if s_value > 0 and t_value > 0 and christof_value > 0:
        status = 'Positive'
    elif s_value > 0 and t_value < 0 and christof_value > 0:
        status = 's+ t- christof+'
    elif s_value > 0 and t_value > 0 and christof_value < 0:
        status = 's+ t+ christof-'
    elif s_value > 0 and t_value < 0 and christof_value < 0:
        status = 's+ t- christof-'
    elif s_value < 0 and t_value > 0 and christof_value > 0:
        status = 's- t+ christof+'
    elif s_value < 0 and t_value < 0 and christof_value > 0:
        status = 's- t- christof+'
    elif s_value < 0 and t_value > 0 and christof_value < 0:
        status = 's- t+ christof-'
    elif s_value < 0 and t_value < 0 and christof_value < 0:
        status = 'Negative'

    common_trf_labels += f'{trf} ({status})\n'
    
venn.get_label_by_id('111').set_text(common_trf_labels)

sandt = sandt - common_trfs
sandt_labels = ''
for trf in sandt:
    s_value = all_significant_data_s.loc[all_significant_data_s['trf'] == trf, 'logFC'].values[0]  # LogFC value from Set S
    t_value = all_significant_data_t.loc[all_significant_data_t['trf'] == trf, 'logFC'].values[0]  # LogFC value from Set T
    if s_value > 0 and t_value > 0:
        status = 'Positive'
    elif s_value > 0 and t_value < 0:
        status = 's+ t-'
    elif s_value < 0 and t_value > 0:
        status = 's- t+'
    elif s_value < 0 and t_value < 0:
        status = 'Negative'

    sandt_labels += f'{trf} ({status})\n'

venn.get_label_by_id('110').set_text(sandt_labels)
    
sandc = sandc - common_trfs
sandc_labels = ''
for trf in sandc:
    s_value = all_significant_data_s.loc[all_significant_data_s['trf'] == trf, 'logFC'].values[0]  # LogFC value from Set S
    christof_value = christof_data.loc[christof_data['trf'] == trf, 'log2FoldChange'].values[0]  # LogFC value from Christof data
    if s_value > 0 and christof_value > 0:
        status = 'Positive'
    elif s_value > 0 and christof_value < 0:
        status = 's+ christof-'
    elif s_value < 0 and christof_value > 0:
        status = 's- christof+'
    elif s_value < 0 and christof_value < 0:
        status = 'Negative'
    sandc_labels += f'{trf} ({status})\n'

venn.get_label_by_id('101').set_text(sandc_labels)

tandc = tandc - common_trfs
tandc_labels = ''
for trf in tandc:
    t_value = all_significant_data_t.loc[all_significant_data_t['trf'] == trf, 'logFC'].values[0]  # LogFC value from Set T
    christof_value = christof_data.loc[christof_data['trf'] == trf, 'log2FoldChange'].values[0]  # LogFC value from Christof data
    if t_value > 0 and christof_value > 0:
        status = 'Positive'
    elif t_value > 0 and christof_value < 0:
        status = 't+ christof-'
    elif t_value < 0 and christof_value > 0:
        status = 't- christof+'
    elif t_value < 0 and christof_value < 0:
        status = 'Negative'
    tandc_labels += f'{trf} ({status})\n'

venn.get_label_by_id('011').set_text(tandc_labels)  

# Adjust the font size
for label in venn.set_labels:  # Set font size for set labels (Set 1, Set 2, Set 3)
    label.set_fontsize(12)
    
for label in venn.subset_labels:  # Set font size for subset labels (numbers or elements inside circles)
    if label:
        label.set_fontsize(6)  # Adjust font size for the elements   # All sets

plt.title('Venn Diagram of Significant tRFs in the S and T ganglion and Christof Blood tRFs')
plt.show()
#plt.savefig('Venn_diagram_S_T_Christof_Blood.svg', format='svg')
plt.close()



# # #plotting the logfc vs time taken in the S ganglion
# yaxis= [s_4_significant['logFC'] , s_24_significant['logFC'], s_168_significant['logFC']]
# yaxis = [np.log2(np.exp(item)) for sublist in yaxis for item in sublist]
# xaxis = ['4h'] * 33 + ['24h'] * 14 + ['7d'] * 36

# # Convert categorical x-axis labels to numerical values
# x_values = np.array([0 if x == '4h' else 1 if x == '24h' else 2 for x in xaxis])

# # Add jitter to x-values
# jitter = np.random.uniform(-0.1, 0.1, size=len(x_values))  # Adjust jitter range if needed
# x_values_jittered = x_values + jitter

# # Colors list and proteins
# colors = []
# proteins = [s_4_significant['trna'], s_24_significant['trna'], s_168_significant['trna']]
# possible_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'pink', 'brown', 'cyan', 
#                    'magenta', 'grey', 'lime', 'olive', 'teal', 'navy', 'maroon', 'aqua', 'fuchsia', 'silver', 
#                    'gray', 'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'pink', 'brown', 
#                    'cyan', 'magenta', 'grey', 'lime', 'olive', 'teal', 'navy', 'maroon', 'aqua', 'fuchsia', 
#                    'silver', 'gray']

# # trf_type_color_mapping = {
# #     'i-tRF': '#1f77b4',  # Soft blue
# #     '3-tRF': '#2ca02c',  # Soft green
# #     '5-tRF': '#ff7f0e',  # Soft orange
# #     '5-half': '#d62728'  # Soft red
# # }


# #Create a colormap for the proteins
# protein_color_dict = dict()
# for protein_series in proteins:
#     for protein in protein_series:
#         if protein not in protein_color_dict:
#             protein_color_dict[protein] = possible_colors.pop()
#         colors.append(protein_color_dict[protein])

# # Scatter plot with jittered x-axis values
# plt.scatter(x_values_jittered, yaxis, c=colors)
# plt.xlabel('Time taken')
# plt.ylabel('logFC')

# # Creating the legend
# color_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=protein_color_dict[protein], markersize=8) 
#                  for protein in protein_color_dict]
# plt.legend(color_handles, protein_color_dict.keys(), title="Type of Trf", bbox_to_anchor=(1.05, 1), loc='best')

# plt.title('logFC vs Time taken in the S ganglion')
# plt.xticks([0, 1, 2], ['4h', '24h', '7d'])

# plt.show()
# plt.close()












