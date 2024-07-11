import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

# Perform PCA
pca = PCA(n_components=4, whiten=True)
pca.fit(pcaData1.iloc[:, :-1])  # Exclude the last column (Sample_ID or other non-numeric)

# Transform the data
pca_transformed = pca.transform(pcaData1.iloc[:, :-1])

# Plot PCA results
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot first vs. second principal component
axs[0].scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=pcaData1['Time_taken'], cmap='viridis')
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2')
axs[0].set_title('PCA: PC1 vs PC2')

# Plot third vs. fourth principal component
axs[1].scatter(pca_transformed[:, 2], pca_transformed[:, 3], c=pcaData1['Time_taken'], cmap='viridis')
axs[1].set_xlabel('PC3')
axs[1].set_ylabel('PC4')
axs[1].set_title('PCA: PC3 vs PC4')

plt.colorbar(axs[1].scatter(pca_transformed[:, 2], pca_transformed[:, 3], c=pcaData1['Time_taken'], cmap='viridis'), ax=axs[1], label="Time taken")
plt.colorbar(axs[0].scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=pcaData1['Time_taken'], cmap='viridis'), ax=axs[0], label="Time taken")

