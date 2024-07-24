import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

#######Data Preprocessing########

# Read in the data
ColData = pd.read_csv('Samples_info.csv', header=0, index_col=0)
CountsDataFrame = pd.read_csv('tRNA_Exclusive_Combined_data.csv', header=0, index_col=0)

# Filter out rows where sum > 2
CountsDataFrame = CountsDataFrame[CountsDataFrame.sum(axis=1) > 2]

# Transpose the data frame for PCA
Hanani_proccessed_data = CountsDataFrame.T

# Add sample IDs as a column
Hanani_proccessed_data['Sample_ID'] = Hanani_proccessed_data.index

# Merge with sample information
Hanani_proccessed_data = pd.merge(Hanani_proccessed_data, ColData, left_on='Sample_ID', right_index=True)

# Set index to Sample_ID
Hanani_proccessed_data.set_index('Sample_ID', inplace=True)

#drop column Ester_4h_LPS_F_S_S3_R1_001.flexbar_q.fastq (problematic column)
Hanani_proccessed_data = Hanani_proccessed_data.drop('Ester_4h_LPS_F_S_S3_R1_001.flexbar_q.fastq')

# Remove non-numeric columns for PCA
non_numeric_columns = ['Time_taken', 'Treatment', 'Sex', 'Place_taken', 'Sample_num']
numeric_columns = [col for col in Hanani_proccessed_data.columns if col not in non_numeric_columns]


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
def plot_cluster(x, y, title, xlabel, ylabel, labels, data=Hanani_proccessed_data, jitter=True):
    # Ensure the DataFrame is being used properly
    if jitter:
        jitter_strength = 0.1
        jittered_x = data.iloc[:, x] + np.random.uniform(-jitter_strength, jitter_strength, data.shape[0])
        jittered_y = data.iloc[:, y] + np.random.uniform(-jitter_strength, jitter_strength, data.shape[0])
    else:
        jittered_x = data.iloc[:, x]
        jittered_y = data.iloc[:, y]

    # Plotting
    plt.figure(figsize=(18, 8))
    plt.scatter(jittered_x, jittered_y, c=labels, cmap='viridis', s=100, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def print_cluster_info(data, labels, cluster_column):
    for cluster in np.unique(labels):
        print('Cluster: ', cluster)
        print('num of samples: ', data[data[cluster_column] == cluster].shape[0])
        print('LPS: ', data[(data[cluster_column] == cluster) & (data['Treatment'] == 1)].shape[0])
        print('ctrl: ', data[(data[cluster_column] == cluster) & (data['Treatment'] == 0)].shape[0])
        print('4h: ', data[(data[cluster_column] == cluster) & (data['Time_taken'] == 4)].shape[0])
        print('24h: ', data[(data[cluster_column] == cluster) & (data['Time_taken'] == 24)].shape[0])
        print('7d: ', data[(data[cluster_column] == cluster) & (data['Time_taken'] == 7 * 24)].shape[0])


#######K means clustering########

from sklearn.cluster import KMeans

# Mapping dictionary
time_mapping = {'4h': 4,'24h': 24, '7d': 7 * 24 }
Sex_mapping = {'M': 0, 'F': 1}
Treatment_mapping = {'CNT': 0, 'LPS': 1}
Place_taken_mapping = {'S': 0, 'T': 1}

# Convert the column using map
Hanani_proccessed_data['Time_taken'] = Hanani_proccessed_data['Time_taken'].map(time_mapping)
Hanani_proccessed_data['Sex'] = Hanani_proccessed_data['Sex'].map(Sex_mapping)
Hanani_proccessed_data['Treatment'] = Hanani_proccessed_data['Treatment'].map(Treatment_mapping)
Hanani_proccessed_data['Place_taken'] = Hanani_proccessed_data['Place_taken'].map(Place_taken_mapping)
Hanani_proccessed_data['Sample_num'] = Hanani_proccessed_data['Sample_num'].str.replace('S', '')

# Apply K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(Hanani_proccessed_data)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
Hanani_proccessed_data['kmeans_cluster'] = labels

# Plotting K-means clustering results
#plot_cluster(0, 1, 'K-means Clustering', 'PC1', 'PC2', labels=Hanani_proccessed_data['kmeans_cluster'])
#print_cluster_info(Hanani_proccessed_data, labels, 'kmeans_cluster')

#######Spectral Clustering########
from sklearn.cluster import SpectralClustering

# Initialize SpectralClustering
spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0)
spectral_labels = spectral.fit_predict(Hanani_proccessed_data[numeric_columns])

# Add SpectralClustering labels to the data
Hanani_proccessed_data['Spectral_Cluster'] = spectral_labels

# Plotting Spectral Clustering results
plot_cluster(Hanani_proccessed_data['Spectral_Cluster'],Hanani_proccessed_data['Time_taken'], 'Spectral Clustering', 'Spectral Cluster', 'Time Taken', labels=Hanani_proccessed_data['Treatment'])
print_cluster_info(Hanani_proccessed_data, Hanani_proccessed_data['Spectral_Cluster'], 'Spectral_Cluster')

#######Agglomerative Clustering########
from sklearn.cluster import AgglomerativeClustering

# Initialize AgglomerativeClustering
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')

# Fit the model
agg_labels = agg_clustering.fit_predict(Hanani_proccessed_data[numeric_columns])

# Add AgglomerativeClustering labels to the data
Hanani_proccessed_data['Agglomerative_Cluster'] = agg_labels

# Plotting Agglomerative Clustering results
plot_cluster(Hanani_proccessed_data['Agglomerative_Cluster'],Hanani_proccessed_data['Time_taken'], 'Agglomerative Clustering', 'Agglomerative Cluster', 'Time Taken', labels=Hanani_proccessed_data['Treatment'])
print_cluster_info(Hanani_proccessed_data, Hanani_proccessed_data['Agglomerative_Cluster'], 'Agglomerative_Cluster')

#######Hierarchical Clustering########
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.cluster.hierarchy as sch

# Compute the linkage matrix
Z = linkage(Hanani_proccessed_data[numeric_columns], method='ward')

# Form clusters
hierarchical_labels = fcluster(Z, t=3, criterion='maxclust')

# Add Hierarchical labels to the data
Hanani_proccessed_data['Hierarchical_Cluster'] = hierarchical_labels

# Plotting Hierarchical Clustering results
plot_cluster(Hanani_proccessed_data['Hierarchical_Cluster'],Hanani_proccessed_data['Time_taken'], 'Hierarchical Clustering', 'Hierarchical Cluster', 'Time Taken', labels=Hanani_proccessed_data['Treatment'])
print_cluster_info(Hanani_proccessed_data, Hanani_proccessed_data['Hierarchical_Cluster'], 'Hierarchical_Cluster')
