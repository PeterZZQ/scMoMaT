rm(list = ls())
gc()
setwd("~/Dropbox/Research/Projects/CFRM/CFRM")
# library(rliger)
library(liger)
library(Matrix)
library(patchwork)

# Read in the data: 2 batches, 3 clusters
dir <- './data/simulated/'

paths <- c('2b3c_sigma0.1_b1_1/', '2b3c_sigma0.1_b1_2/', '2b3c_sigma0.2_b1_1/', '2b3c_sigma0.2_b1_2/', '2b3c_sigma0.3_b1_1/', '2b3c_sigma0.3_b1_2/',
           '2b4c_sigma0.1_b1_1/', '2b4c_sigma0.1_b1_2/', '2b4c_sigma0.2_b1_1/', '2b4c_sigma0.2_b1_2/', '2b4c_sigma0.3_b1_1/', '2b4c_sigma0.3_b1_2/',
           '2b5c_sigma0.1_b1_1/', '2b5c_sigma0.1_b1_2/', '2b5c_sigma0.2_b1_1/', '2b5c_sigma0.2_b1_2/', '2b5c_sigma0.3_b1_1/', '2b5c_sigma0.3_b1_2/'
           )

paths <- c('2b3c_sigma0.1_b2_1/', '2b3c_sigma0.1_b2_2/', '2b3c_sigma0.2_b2_1/', '2b3c_sigma0.2_b2_2/', '2b3c_sigma0.3_b2_1/', '2b3c_sigma0.3_b2_2/',
           '2b4c_sigma0.1_b2_1/', '2b4c_sigma0.1_b2_2/', '2b4c_sigma0.2_b2_1/', '2b4c_sigma0.2_b2_2/', '2b4c_sigma0.3_b2_1/', '2b4c_sigma0.3_b2_2/',
           '2b5c_sigma0.1_b2_1/', '2b5c_sigma0.1_b2_2/', '2b5c_sigma0.2_b2_1/', '2b5c_sigma0.2_b2_2/', '2b5c_sigma0.3_b2_1/', '2b5c_sigma0.3_b2_2/'
)

paths <- c('2b3c_sigma0.4_b1_1/', '2b3c_sigma0.4_b1_2/', '2b4c_sigma0.4_b1_1/', '2b4c_sigma0.4_b1_2/', '2b5c_sigma0.4_b1_1/', '2b5c_sigma0.4_b1_2/')
paths <- c('2b3c_sigma0.5_b1_1/', '2b3c_sigma0.5_b1_2/', '2b4c_sigma0.5_b1_1/', '2b4c_sigma0.5_b1_2/', '2b5c_sigma0.5_b1_1/', '2b5c_sigma0.5_b1_2/')


quant = F


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
    
    # test Horizontal
    print("Horizontal")
    ifnb_liger <- rliger::createLiger(list(rna1 = counts_rna1, rna2 = counts_rna2))
    ifnb_liger <- rliger::normalize(ifnb_liger)
    ifnb_liger <- rliger::selectGenes(ifnb_liger)
    ifnb_liger <- rliger::scaleNotCenter(ifnb_liger)

    # online ver: http://htmlpreview.github.io/?https://github.com/welch-lab/liger/blob/master/vignettes/online_iNMF_tutorial.html
    # ifnb_liger = online_iNMF(ifnb_liger, k = 20, miniBatch_size = 5000, max.epochs = 5)
    # unshared ver: http://htmlpreview.github.io/?https://github.com/welch-lab/liger/blob/master/vignettes/UINMF_vignette.html
    # ifnb_liger <- optimizeALS(ifnb_liger, k=30, use.unshared = TRUE)
    # original
    ifnb_liger <- rliger::optimizeALS(ifnb_liger, k = num_clust)

    # quantile normalization
    print("Writing results...")
    if(quant == T){
        ifnb_liger <- quantile_norm(ifnb_liger)
        write.csv(ifnb_liger@clusters, paste0("./test/results_multi_liger_quantile/hori/", substr(path, 1, nchar(path)-1), "_clust_id.csv"))
    }else{
        H1 <- ifnb_liger@H$rna1
        H2 <- ifnb_liger@H$rna2
        cluster1 <- cbind(row.names(H1), max.col(H1, 'first'))
        cluster2 <- cbind(row.names(H2), max.col(H2, 'first'))
        cluster <- rbind(cluster1, cluster2)
        row.names(cluster) <-cluster[,1]
        cluster <- cluster[,-1]
        write.csv(cluster, paste0("./test/results_multi_liger/hori/", substr(path, 1, nchar(path)-1), "_clust_id.csv"))
    }


    # test Diagonal
    print("Diagonal")
    ifnb_liger <- rliger::createLiger(list(rna1 = counts_rna1, atac2 = as.data.frame(t(as.matrix(gene_act)) %*% as.matrix(counts_atac2))))
    ifnb_liger <- rliger::normalize(ifnb_liger)
    ifnb_liger <- rliger::selectGenes(ifnb_liger)
    ifnb_liger <- rliger::scaleNotCenter(ifnb_liger)
    ifnb_liger <- rliger::optimizeALS(ifnb_liger, k = num_clust)
    # quantile normalization
    print("Writing results...")
    if(quant == T){
        ifnb_liger <- quantile_norm(ifnb_liger)

        write.csv(ifnb_liger@clusters, paste0("./test/results_multi_liger_quantile/diag/", substr(path, 1, nchar(path)-1), "_clust_id.csv"))
    }else{
        H1 <- ifnb_liger@H$rna1
        H2 <- ifnb_liger@H$atac2
        cluster1 <- cbind(row.names(H1), max.col(H1, 'first'))
        cluster2 <- cbind(row.names(H2), max.col(H2, 'first'))
        cluster <- rbind(cluster1, cluster2)
        row.names(cluster) <-cluster[,1]
        cluster <- cluster[,-1]
        write.csv(cluster, paste0("./test/results_multi_liger/diag/", substr(path, 1, nchar(path)-1), "_clust_id.csv"))
    }
    # test Vertical, check unshared ver
    # TODO:?
    # test Multi4, check unshared ver
    print("Multi4")
    ifnb_liger <- rliger::createLiger(list(rna_atac1 = rbind(counts_rna1, counts_atac1), rna_atac2 = rbind(counts_rna2, counts_atac2)))
    ifnb_liger <- rliger::normalize(ifnb_liger)
    ifnb_liger <- rliger::selectGenes(ifnb_liger)
    ifnb_liger <- rliger::scaleNotCenter(ifnb_liger)
    ifnb_liger <- rliger::optimizeALS(ifnb_liger, k = num_clust)
    if(quant == T){
        ifnb_liger <- quantile_norm(ifnb_liger)
        write.csv(ifnb_liger@clusters, paste0("./test/results_multi_liger_quantile/multi4/", substr(path, 1, nchar(path)-1), "_clust_id.csv"))
    }else{
        H1 <- ifnb_liger@H$rna_atac1
        H2 <- ifnb_liger@H$rna_atac2
        cluster1 <- cbind(row.names(H1), max.col(H1, 'first'))
        cluster2 <- cbind(row.names(H2), max.col(H2, 'first'))
        cluster <- rbind(cluster1, cluster2)
        row.names(cluster) <-cluster[,1]
        cluster <- cluster[,-1]
        write.csv(cluster, paste0("./test/results_multi_liger/multi4/", substr(path, 1, nchar(path)-1), "_clust_id.csv"))
    }

    # test Multi3, check unshared ver
    # library(devtools)
    # install_github("welch-lab/liger", ref = "U_algorithm")
    print("Multi3")
    ifnb_liger <- liger::createLiger(list(rna_atac1 = rbind(counts_rna1, counts_atac1), rna2 = counts_rna2))
    ifnb_liger <- liger::normalize(ifnb_liger)
    ifnb_liger <- liger::selectGenes(ifnb_liger, unshared = TRUE, unshared.datasets = list(1), unshared.thresh= 0.0)
    ifnb_liger <- liger::scaleNotCenter(ifnb_liger)
    ifnb_liger <- liger::optimizeALS(ifnb_liger, k = num_clust)
    print("Writing results...")
    if(quant == T){
        ifnb_liger <- quantile_norm(ifnb_liger)
        write.csv(ifnb_liger@clusters, paste0("./test/results_multi_liger_quantile/multi3/", substr(path, 1, nchar(path)-1), "_clust_id.csv"))
    }else{
        H1 <- ifnb_liger@H$rna_atac1
        H2 <- ifnb_liger@H$rna2
        cluster1 <- cbind(row.names(H1), max.col(H1, 'first'))
        cluster2 <- cbind(row.names(H2), max.col(H2, 'first'))
        cluster <- rbind(cluster1, cluster2)
        row.names(cluster) <-cluster[,1]
        cluster <- cluster[,-1]
        write.csv(cluster, paste0("./test/results_multi_liger/multi3/", substr(path, 1, nchar(path)-1), "_clust_id.csv"))
    }
}

