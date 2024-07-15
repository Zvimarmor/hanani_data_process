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
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
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
kmeans = KMeans(n_clusters=3)
kmeans.fit(Hanani_proccessed_data)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
Hanani_proccessed_data['Cluster'] = labels

# Add jitter to the data points
jitter_strength = 0.1
jittered_time_taken = Hanani_proccessed_data['Place_taken'] + np.random.uniform(-jitter_strength, jitter_strength, Hanani_proccessed_data.shape[0])
jittered_treatment = Hanani_proccessed_data['Cluster'] + np.random.uniform(-jitter_strength, jitter_strength, Hanani_proccessed_data.shape[0])

# Plotting
plt.figure(figsize=(18, 8))
plt.scatter(jittered_time_taken, jittered_treatment, c=Hanani_proccessed_data['Treatment'], cmap='viridis', s=100, alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=50, linewidths=2)
plt.xlabel('Time_taken')
plt.ylabel('Cluster')
plt.title('K-means Clustering with Jitter')
plt.show()


# # Plot first vs. second principal component from PCA
# plot_with_legend(pca_transformed, 0, 1, Hanani_proccessed_data['Combined_Label'], 'PCA: PC1 vs PC2', 'PC1', 'PC2', label_mapping)

# # Plot third vs. fourth principal component from PCA
# plot_with_legend(pca_transformed, 2, 3, Hanani_proccessed_data['Combined_Label'], 'PCA: PC3 vs PC4', 'PC3', 'PC4', label_mapping)

# # Plot t-SNE results
# plot_with_legend(tsne_transformed, 0, 1, Hanani_proccessed_data['Combined_Label'], 't-SNE', 't-SNE 1', 't-SNE 2', label_mapping)

# # Plot LDA results
# plot_with_legend(lda_transformed, 0, 1, Hanani_proccessed_data['Combined_Label'], 'LDA', 'LDA 1', 'LDA 2', label_mapping)

# Plot K-means clustering results
# plot_with_legend(Hanani_proccessed_data[numeric_columns].values, 0, 1, Hanani_proccessed_data['Cluster'], 'K-means Clustering', 'PC1', 'PC2', {})