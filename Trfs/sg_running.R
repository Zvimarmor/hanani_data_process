library(edgeR)
library(ggplot2)
library(biomaRt)
library(matrixStats)
library(MASS)
setwd("/Users/zvimarmor/university_second_year/hanani_data_process")

ColData <- read.csv('Samples_info.csv', header = TRUE, row.names = 1)
CountsDataFrame <- read.csv("tRNA_data.csv", header = TRUE, row.names = 1)

CountsDataFrame<-CountsDataFrame[rowSums(CountsDataFrame)>2,]
# Assign temporary variable names for counts data and processed column data
tmp1 <- CountsDataFrame
pcaData1 <- as.data.frame(t(tmp1))

# Add sample IDs as a column
pcaData1$Sample_ID <- rownames(pcaData1) # change 'Sample_ID' to the name of your sample ID column
ColData$Sample_ID <- rownames(ColData)

pcaData1 <- merge(pcaData1, ColData, by = "Sample_ID")

# Set row names to sample IDs
rownames(pcaData1) <- pcaData1$Sample_ID


# Perform PCA on the processed data
pca_res1 <- prcomp(pcaData1[2:ncol(tmp1)], scale = TRUE)

# Specify the parameter for labeling the PCA plot
c <- "Treatment"

# Plot PCA results for the first two principal components
autoplot(pca_res1, x = 1, y = 2, data = pcaData1, colour = c, label = F, size = 3)

# Plot PCA results for the third and fourth principal components
autoplot(pca_res1, x = 3, y = 4, data = pcaData1, colour = c, label = F, size = 3)

# Assign counts data to a variable
cts <- CountsDataFrame

# Assign column data to a variable
cold1 <- ColData
print(cold1)

# Subset column data to match the sample IDs in counts data
cold1 <- subset(cold1, cold1$Sample_ID %in% colnames(cts))

# Order counts data based on sample IDs in column data
cts <- cts[, as.character(cold1$Sample_ID)]
cts <- cts[, order(cold1$Sample_ID)]
# Order column data based on sample IDs
cold1 <- cold1[order(cold1$Sample_ID),]
# Check if row names of column data match column names of counts data
nrow(cold1) == sum(cold1$Sample_ID == colnames(cts))

# Create DGEList object for differential expression analysis
y <- DGEList(counts = cts, group = cold1$Treatment) # change 'condition' to the name of your condition column

# Filter genes by expression
keep <- filterByExpr(y)
y <- y[keep, , keep.lib.sizes = FALSE]

# Update library sizes
y$samples$lib.size <- colSums(y$counts)

# Calculate normalization factors
y <- calcNormFactors(y)

# Create normalized counts matrix
cts1 <- as.data.frame(cpm(y, log = FALSE))
write.csv(cts1, file = "cpm.csv", row.names = TRUE)

# Create model matrix for differential expression analysis
dsgn <- model.matrix( ~Time_taken+Sex+Place_taken+Treatment, data = cold1)

# Estimate dispersions
y <- estimateDisp(y, dsgn, robust = TRUE)

# If desired, run the next code to see if the different coefficients correlate with each other
logFC <- predFC(y, dsgn, prior.count = 1, dispersion = 0.05)
cor(logFC)
plotBCV(y)
# Display the head of the design matrix
head(dsgn)
# Change the coefficient to the coefficient that interests you (the number of the column in the design matrix)
fit <- glmQLFit(y, dsgn, robust = TRUE)
lrt1 <- glmLRT(fit, coef = 2)
# Extract top differentially expressed genes
sgGens <- as.data.frame(topTags(lrt1, adjust.method = 'fdr', n = nrow(cts1)))
sgGens$transcript <- rownames(sgGens)
write.csv(sgGens, file = "sg_genes_all.csv", row.names = FALSE)
