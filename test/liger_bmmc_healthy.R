rm(list =ls())
gc()
library(Signac)
library(Seurat)
library(GenomeInfoDb)
library(ggplot2)
library(patchwork)
library(Matrix)
# setwd("/localscratch/ziqi/CFRM/test/")


################################################
#
# Using Raw data
#
################################################


# Read in the raw data
subsampling <- 5
if(subsampling == 1){
  path <- "../data/real/diag/healthy_hema/topgenes_1000/BMMC/"
  result_path <- "bmmc_healthyhema_1000/liger/"
  counts.atac <- readMM(paste0(path, "RxC1.mtx"))
  rownames(counts.atac) <- read.table(paste0(path, "regions.txt"), header = F, sep = ",")[[1]]
  colnames(counts.atac) <- rownames(read.table(paste0(path, "meta_c1.csv"), header = T, row.names = 1, sep = ","))
  counts.rna <- readMM(paste0(path, "GxC2.mtx"))
  rownames(counts.rna) <- read.table(paste0(path, "genes.txt"), header = F, sep = ",")[[1]]
  colnames(counts.rna) <- rownames(read.table(paste0(path, "meta_c2.csv"), header = T, row.names = 1, sep = ","))
  gene2region <- readMM(paste0(path, "GxR.mtx"))
  rownames(gene2region) <- rownames(counts.rna)
  colnames(gene2region) <- rownames(counts.atac)  
}else{
  path <- "../data/real/diag/healthy_hema/topgenes_1000/BMMC/"
  result_path <- paste0("bmmc_healthyhema_1000/subsample_", subsampling, "/liger/")
  counts.atac <- readMM(paste0(path, "RxC1.mtx"))
  rownames(counts.atac) <- read.table(paste0(path, "regions.txt"), header = F, sep = ",")[[1]]
  colnames(counts.atac) <- rownames(read.table(paste0(path, "meta_c1.csv"), header = T, row.names = 1, sep = ","))
  counts.rna <- readMM(paste0(path, "GxC2.mtx"))
  rownames(counts.rna) <- read.table(paste0(path, "genes.txt"), header = F, sep = ",")[[1]]
  colnames(counts.rna) <- rownames(read.table(paste0(path, "meta_c2.csv"), header = T, row.names = 1, sep = ","))
  gene2region <- readMM(paste0(path, "GxR.mtx"))
  rownames(gene2region) <- rownames(counts.rna)
  colnames(gene2region) <- rownames(counts.atac)  
  
  # subsampling
  counts.atac <- counts.atac[,seq(1, dim(counts.atac)[2], subsampling)]
  counts.rna <- counts.ran[,seq(1, dim(counts.rna)[2], subsampling)]
}



activity.matrix <- gene2region %*% counts.atac

ifnb_liger <- rliger::createLiger(list(atac1 = activity.matrix, rna2 = counts.rna), remove.missing = FALSE)
ifnb_liger <- rliger::normalize(ifnb_liger, verbose = TRUE)
# If use 2000 genes, here Liger filter it into 711 genes, for 1000 genes, Liger filter it into 410 genes. Here we select all the genes instead.
# ifnb_liger <- rliger::selectGenes(ifnb_liger, datasets.use = 2)
ifnb_liger@var.genes <- rownames(counts.rna)
ifnb_liger <- rliger::scaleNotCenter(ifnb_liger, remove.missing = FALSE)

num_clust <- 30
ifnb_liger <- rliger::optimizeALS(ifnb_liger, k = num_clust)
# quantile normalization
print("Writing results...")

ifnb_liger <- rliger::quantile_norm(ifnb_liger)
H1 <- ifnb_liger@H$atac1
H2 <- ifnb_liger@H$rna2

H.norm1 <- ifnb_liger@H.norm[1:dim(H1)[1],]
H.norm2 <- ifnb_liger@H.norm[(dim(H1)[1] + 1):(dim(H1)[1] + dim(H2)[1]),]
write.csv(H.norm1, paste0(result_path, "liger_c1.csv"))
write.csv(H.norm2, paste0(result_path, "liger_c2.csv"))
