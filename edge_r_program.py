import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr

# Activate the automatic conversion of pandas dataframes to R dataframes
pandas2ri.activate()

# Import edgeR and other required R packages
edgeR = importr('edgeR')
ggplot2 = importr('ggplot2')
matrixStats = importr('matrixStats')
MASS = importr('MASS')

# Read in the data
df = pd.read_csv('Hanani_proccessed_data_S.csv', index_col=0)

# Assuming the gene expression columns are all columns except the last few columns containing sample info
gene_expression_cols = df.columns[:-5]  # Adjust the range according to your dataset

# Filter rows with counts > 2
df = df[df[gene_expression_cols].sum(axis=1) > 2]

# # Perform PCA on gene expression data
# pca = PCA(n_components=4)
# pca_results = pca.fit_transform(df[gene_expression_cols])

# # Add PCA results to the DataFrame
# df['PC1'] = pca_results[:, 0]
# df['PC2'] = pca_results[:, 1]
# df['PC3'] = pca_results[:, 2]
# df['PC4'] = pca_results[:, 3]

# # Plot PCA results
# plt.figure(figsize=(14, 7))

# plt.subplot(1, 2, 1)
# sns.scatterplot(x='PC1', y='PC2', hue='Time_taken', data=df)
# plt.title('PCA: PC1 vs PC2')

# plt.subplot(1, 2, 2)
# sns.scatterplot(x='PC3', y='PC4', hue='Time_taken', data=df)
# plt.title('PCA: PC3 vs PC4')

# plt.tight_layout()
# plt.show()

# Filter genes based on quantile and mean and median values
# to_remove = [g for g in gene_expression_cols if df[g].quantile(0.85) < df[g].mean()]
# to_remove += [g for g in gene_expression_cols if df[g].median() < 10]
# df_filtered = df.drop(columns=to_remove)
df_filtered = df

# Create DGEList object for differential expression analysis using edgeR
counts = pandas2ri.py2rpy(df_filtered[gene_expression_cols])
coldata = pandas2ri.py2rpy(df_filtered[['Treatment', 'Time_taken', 'Sex', 'Place_taken']])

dge = edgeR.DGEList(counts=counts, group=coldata.rx2('Treatment'))

# Calculate normalization factors
dge = edgeR.calcNormFactors(dge)

# Create model matrix for differential expression analysis
design = r['model.matrix']('~Time_taken+Sex+Place_taken+Treatment', data=coldata)

# Estimate dispersions
dge = edgeR.estimateDisp(dge, design, robust=True)

# Fit the model and perform differential expression analysis
fit = edgeR.glmQLFit(dge, design, robust=True)
lrt = edgeR.glmLRT(fit, coef=4)  # Change coef to the correct column index

# Extract top differentially expressed genes
top_tags = edgeR.topTags(lrt, n=float('inf'), adjust_method='fdr')
top_genes = pandas2ri.rpy2py(top_tags.rx2('table'))

# Save results to CSV
top_genes.to_csv("differential_expression_results_S.csv", index=True)

# # Example plotting specific gene expression
# trf = 'tRF-29-PSQP4PW3FJFL'  # Change to your specific gene of interest
# df_trf = df_filtered[['Sample_ID', trf, 'Time_taken', 'Treatment', 'Place_taken']]
# df_trf = df_trf.melt(id_vars=['Sample_ID', 'Time_taken', 'Treatment', 'Place_taken'], value_vars=[trf], var_name='Gene', value_name='CPM')

# plt.figure(figsize=(10, 6))
# sns.boxplot(x='Time_taken', y='CPM', hue='Treatment', data=df_trf)
# sns.stripplot(x='Time_taken', y='CPM', hue='Treatment', data=df_trf, dodge=True, marker='o', alpha=0.7, palette='dark:k')
# plt.yscale('log')
# plt.title(f'CPM of {trf}')
# plt.show()
