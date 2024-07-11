import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Read in the data
ColData = pd.read_csv('Samples_info.csv', header=0, index_col=0)
CountsDataFrame = pd.read_csv('tRNA_Exclusive_Combined_data.csv', header=0, index_col=0)

# Filter out rows where sum > 2
CountsDataFrame = CountsDataFrame[CountsDataFrame.sum(axis=1) > 2]

# Transpose the data frame for PCA
pcaData1 = CountsDataFrame.transpose()

# Add sample IDs as a column
pcaData1['Sample_ID'] = pcaData1.index

# Merge with sample information
pcaData1 = pd.merge(pcaData1, ColData, left_on='Sample_ID', right_index=True)

# Set index to Sample_ID
pcaData1.set_index('Sample_ID', inplace=True)

# Remove non-numeric columns for PCA
non_numeric_columns = ['Time_taken','Treatment','Sex','Place_taken','Sample_num']  # Add other non-numeric column names here if needed
numeric_columns = [col for col in pcaData1.columns if col not in non_numeric_columns]

# Perform PCA
pca = PCA(n_components=4, whiten=True)
pca_transformed = pca.fit_transform(pcaData1[numeric_columns])

# Perform t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_transformed = tsne.fit_transform(pcaData1[numeric_columns])

# Encode the 'Time_taken' column to numeric labels
label_encoder = LabelEncoder()
pcaData1['Time_taken_encoded'] = label_encoder.fit_transform(pcaData1['Time_taken'])

# Plot PCA results
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot first vs. second principal component from PCA
scatter1 = axs[0].scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=pcaData1['Time_taken_encoded'], cmap='viridis')
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2')
axs[0].set_title('PCA: PC1 vs PC2')
plt.colorbar(scatter1, ax=axs[0], label="Time taken")

# Plot third vs. fourth principal component from PCA
scatter2 = axs[1].scatter(pca_transformed[:, 2], pca_transformed[:, 3], c=pcaData1['Time_taken_encoded'], cmap='viridis')
axs[1].set_xlabel('PC3')
axs[1].set_ylabel('PC4')
axs[1].set_title('PCA: PC3 vs PC4')
plt.colorbar(scatter2, ax=axs[1], label="Time taken")

# Plot t-SNE results
scatter3 = axs[2].scatter(tsne_transformed[:, 0], tsne_transformed[:, 1], c=pcaData1['Time_taken_encoded'], cmap='viridis')
axs[2].set_xlabel('t-SNE Component 1')
axs[2].set_ylabel('t-SNE Component 2')
axs[2].set_title('t-SNE')
plt.colorbar(scatter3, ax=axs[2], label="Time taken")

plt.tight_layout()
plt.show()
