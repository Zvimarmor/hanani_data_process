# Data Analysis Project

This project involves the preprocessing, dimensionality reduction, and clustering of tRNA data. The main steps include reading in the data, filtering, mapping non-numeric columns to numeric values, and performing clustering using the K-means algorithm.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Data Preprocessing](#data-preprocessing)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Clustering](#clustering)
- [Plotting](#plotting)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Prerequisites
- pandas
- numpy
- scikit-learn
- matplotlib

## Data Preprocessing

The preprocessing steps involve:
1. Reading in the data from CSV files.
2. Filtering out rows where the sum of counts is greater than 2.
3. Transposing the data for Principal Component Analysis (PCA).
4. Adding sample IDs as a column and merging with additional sample information.
5. Mapping non-numeric columns to numeric values for PCA and clustering.

## Dimensionality Reduction

Dimensionality reduction techniques used in this project include:
- Principal Component Analysis (PCA)
- t-distributed Stochastic Neighbor Embedding (t-SNE)
- Linear Discriminant Analysis (LDA)

## Clustering

Clustering is performed using the K-means algorithm. The K-means algorithm clusters the data into a specified number of clusters based on the numeric columns.

## Plotting

Various plots are created to visualize the results of PCA, t-SNE, LDA, and K-means clustering. The plotting function `plot_with_legend` is used to create scatter plots with legends.


## Acknowledgements

This project uses data from `Samples_info.csv` and `tRNA_Exclusive_Combined_data.csv`. 
