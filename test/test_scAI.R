# Install packages
# devtools::install_github("sqjin/scAI")
# https://htmlpreview.github.io/?https://github.com/sqjin/scAI/blob/master/examples/walkthrough_simulation_dataset8.html

# load  
rm(list = ls())
gc()
setwd("/Users/zzhang834/Dropbox/Research/Projects/CFRM/CFRM")
library(scAI)
library(dplyr)
library(cowplot)
library(ggplot2)


# Read in the data: 2 batches, 3 clusters
dir <- './data/simulated/'

paths <- c('2b3c_sigma0.1_b1_1/', '2b3c_sigma0.1_b1_2/', '2b3c_sigma0.2_b1_1/', '2b3c_sigma0.2_b1_2/', '2b3c_sigma0.3_b1_1/', '2b3c_sigma0.3_b1_2/','2b3c_sigma0.4_b1_1/','2b3c_sigma0.4_b1_2/',
           '2b4c_sigma0.1_b1_1/', '2b4c_sigma0.1_b1_2/', '2b4c_sigma0.2_b1_1/', '2b4c_sigma0.2_b1_2/', '2b4c_sigma0.3_b1_1/', '2b4c_sigma0.3_b1_2/','2b4c_sigma0.4_b1_1/', '2b4c_sigma0.4_b1_2/',
           '2b5c_sigma0.1_b1_1/', '2b5c_sigma0.1_b1_2/', '2b5c_sigma0.2_b1_1/', '2b5c_sigma0.2_b1_2/', '2b5c_sigma0.3_b1_1/', '2b5c_sigma0.3_b1_2/','2b5c_sigma0.4_b1_1/', '2b5c_sigma0.4_b1_2/'
           )

paths <- c('2b3c_sigma0.1_b2_1/', '2b3c_sigma0.1_b2_2/', '2b3c_sigma0.2_b2_1/', '2b3c_sigma0.2_b2_2/', '2b3c_sigma0.3_b2_1/', '2b3c_sigma0.3_b2_2/',
           '2b4c_sigma0.1_b2_1/', '2b4c_sigma0.1_b2_2/', '2b4c_sigma0.2_b2_1/', '2b4c_sigma0.2_b2_2/', '2b4c_sigma0.3_b2_1/', '2b4c_sigma0.3_b2_2/',
           '2b5c_sigma0.1_b2_1/', '2b5c_sigma0.1_b2_2/', '2b5c_sigma0.2_b2_1/', '2b5c_sigma0.2_b2_2/', '2b5c_sigma0.3_b2_1/', '2b5c_sigma0.3_b2_2/'
)



for(path in paths){
    print(substr(path, 1, nchar(path)-1))
    num_clust <- strtoi(substr(path, 3,3))

    # data frame
    counts_rna1 <- read.table(file = paste0(dir, path, "GxC1.txt"), header = F, sep = "\t")
    row.names(counts_rna1) <- paste("Gene_", 1:dim(counts_rna1)[1], sep = "")
    colnames(counts_rna1) <- paste("Cell_", 1:dim(counts_rna1)[2], sep = "")
    counts_rna2 <- read.table(file = paste0(dir, path, "GxC2.txt"), header = F, sep = "\t")
    row.names(counts_rna2) <- paste("Gene_", 1:dim(counts_rna2)[1], sep = "")
    colnames(counts_rna2) <- paste("Cell_", (dim(counts_rna1)[2]+1):(dim(counts_rna1)[2] + dim(counts_rna2)[2]), sep = "")
    
    counts_atac1 <- read.table(file = paste0(dir, path, "RxC1.txt"), header = F, sep = "\t")
    rownames(counts_atac1) <- paste("Loc_", 1:dim(counts_atac1)[1], sep = "")
    colnames(counts_atac1) <- paste("Cell_", 1:dim(counts_atac1)[2], sep = "")
    counts_atac2 <- read.table(file = paste0(dir, path, "RxC2.txt"), header = F, sep = "\t")
    rownames(counts_atac2) <- paste("Loc_", 1:dim(counts_atac2)[1], sep = "")
    colnames(counts_atac2) <- paste("Cell_", (dim(counts_atac1)[2]+1):(dim(counts_atac1)[2] + dim(counts_atac2)[2]), sep = "")
    
    gene_act <- read.table(file = paste0(dir, path, "region2gene.txt"), header = F, sep = "\t")
    rownames(gene_act) <- rownames(counts_atac1)
    colnames(gene_act) <- rownames(counts_rna1)
    
    # test vertical
    X <- list(RNA = as.matrix(counts_rna1), ATAC = as.matrix(counts_atac1))
    scAI_outs <- create_scAIobject(raw.data = X)

    # Perform quality control to remove low-quality cells and genes, and normalize the data. Since this is a simulated data, we do not need to normalize the data. Thus we set assay = NULL.
    scAI_outs <- preprocessing(scAI_outs, assay = NULL, minFeatures = 200, minCells = 1, libararyflag = F, logNormalize = F)
    print("running scAI...")
    scAI_outs <- run_scAI(scAI_outs, K = num_clust, nrun = 5)
    print("Finished. Conduct clustering...")
    # We can also identify cell clusters based on the inferred cell loading matrix using Leiden algorithm.
    scAI_outs <- identifyClusters(scAI_outs, resolution = 0.05)
    print("clustered")
    break

    # visualize
    scAI_outs <- reducedDims(scAI_outs, method = "umap")
    cellVisualization(scAI_outs, scAI_outs@embed$umap, color.by = "cluster")

}
