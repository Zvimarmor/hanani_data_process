# Hanani's Data Processing and Analysis Project

This project involves the preprocessing, dimensionality reduction, clustering, and differential expression analysis of tRNA data. The main steps include reading in the data, filtering, mapping non-numeric columns to numeric values, performing PCA, t-SNE, LDA, clustering using the K-means algorithm, and additional analysis using R.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Data Preprocessing](#data-preprocessing)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Clustering](#clustering)
- [Plotting](#plotting)
- [Differential Expression Analysis (R)](#differential-expression-analysis-r)

## Prerequisites

### Python Packages
This project requires the following Python packages:
- pandas
- numpy
- scikit-learn
- matplotlib

### R Packages
This project requires the following R packages:
- edgeR
- ggplot2
- ggfortify
- biomaRt
- matrixStats
- MASS

## Data Preprocessing

The preprocessing steps involve:
1. Reading in the data from CSV files (`Samples_info.csv` and `tRNA_Exclusive_Combined_data.csv`).
2. Filtering out rows where the sum of counts is greater than 2.
3. Transposing the data for Principal Component Analysis (PCA).
4. Adding sample IDs as a column and merging with additional sample information.
5. Mapping non-numeric columns to numeric values for PCA and clustering.

## Dimensionality Reduction

Dimensionality reduction techniques used in this project include:
- **Principal Component Analysis** (PCA): A technique used to reduce the dimensionality of the data while retaining most of the variance.
- **t-distributed Stochastic Neighbor Embedding** (t-SNE): A technique used to visualize high-dimensional data by reducing it to two or three dimensions.
- **Linear Discriminant Analysis** (LDA): A technique used for dimensionality reduction and classification by finding a linear combination of features that best separates two or more classes.

## Clustering

Several clustering techniques are used to group the data based on their features. These techniques help in identifying patterns and grouping similar data points together:

- **K-means Clustering**: Clusters the data into a specified number of clusters based on the numeric columns. This method is straightforward and effective for many types of data.

- **Spectral Clustering**: Uses the spectral properties of the data to cluster points. It is particularly useful for identifying clusters with complex shapes.

- **Agglomerative Clustering**: A type of hierarchical clustering that builds nested clusters by merging or splitting them successively. This method is useful for discovering a hierarchical structure in the data.

- **Hierarchical Clustering**: Uses linkage methods to form a hierarchy of clusters. This method helps in understanding the relationships between clusters at various levels of granularity.

## Plotting

Various plots are created to visualize the results of PCA, t-SNE, LDA, and the different clustering techniques. Scatter plots with legends are used to show the distribution of data points and the clusters they belong to. These plots help in understanding the underlying structure of the data and the effectiveness of the clustering and dimensionality reduction techniques.

## Differential Expression Analysis (R)

Additional analysis is performed using R, including differential expression analysis and plotting. This step involves:
1. Loading and preprocessing the data in R.
2. Performing differential expression analysis using the edgeR package.
3. Visualizing the results with ggplot2 and other plotting libraries.

