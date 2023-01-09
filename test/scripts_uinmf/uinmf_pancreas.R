rm(list = ls())
gc()

# install.packages('devtools')
# library(devtools)
# install_github('welch-lab/liger')
library(rliger)
library(Seurat)
library(stringr)

setwd("/localscratch/ziqi/scMoMaT/test/scripts_uinmf/")
# Read in the count matrix
data_dir <- "../../data/real/hori/Pancreas/"
result_dir <- "../pancreas/LIGER/"

counts.rna.0 <- readMM(paste0(data_dir, 'GxC0.mtx'))
counts.rna.1 <- readMM(paste0(data_dir, 'GxC1.mtx'))
counts.rna.2 <- readMM(paste0(data_dir, 'GxC2.mtx'))
counts.rna.3 <- readMM(paste0(data_dir, 'GxC3.mtx'))
counts.rna.4 <- readMM(paste0(data_dir, 'GxC4.mtx'))
counts.rna.5 <- readMM(paste0(data_dir, 'GxC5.mtx'))
counts.rna.6 <- readMM(paste0(data_dir, 'GxC6.mtx'))
counts.rna.7 <- readMM(paste0(data_dir, 'GxC7.mtx'))

genes <- read.csv(paste0(data_dir, "gene.csv"), header = F)[[1]]
meta.cell.0 <- read.csv(paste0(data_dir, "meta_c0.csv"), header = T, sep = ",", row.names = 1)
meta.cell.1 <- read.csv(paste0(data_dir, "meta_c1.csv"), header = T, sep = ",", row.names = 1)
meta.cell.2 <- read.csv(paste0(data_dir, "meta_c2.csv"), header = T, sep = ",", row.names = 1)
meta.cell.3 <- read.csv(paste0(data_dir, "meta_c3.csv"), header = T, sep = ",", row.names = 1)
meta.cell.4 <- read.csv(paste0(data_dir, "meta_c4.csv"), header = T, sep = ",", row.names = 1)
meta.cell.5 <- read.csv(paste0(data_dir, "meta_c5.csv"), header = T, sep = ",", row.names = 1)
meta.cell.6 <- read.csv(paste0(data_dir, "meta_c6.csv"), header = T, sep = ",", row.names = 1)
meta.cell.7 <- read.csv(paste0(data_dir, "meta_c7.csv"), header = T, sep = ",", row.names = 1)

rownames(meta.cell.0) <- paste("batch_0:cell_", seq(1, dim(meta.cell.0)[1]), sep = "")
rownames(meta.cell.1) <- paste("batch_1:cell_", seq(1, dim(meta.cell.1)[1]), sep = "")
rownames(meta.cell.2) <- paste("batch_2:cell_", seq(1, dim(meta.cell.2)[1]), sep = "")
rownames(meta.cell.3) <- paste("batch_3:cell_", seq(1, dim(meta.cell.3)[1]), sep = "")
rownames(meta.cell.4) <- paste("batch_4:cell_", seq(1, dim(meta.cell.4)[1]), sep = "")
rownames(meta.cell.5) <- paste("batch_5:cell_", seq(1, dim(meta.cell.5)[1]), sep = "")
rownames(meta.cell.6) <- paste("batch_6:cell_", seq(1, dim(meta.cell.6)[1]), sep = "")
rownames(meta.cell.7) <- paste("batch_7:cell_", seq(1, dim(meta.cell.7)[1]), sep = "")

rownames(counts.rna.0) <- genes
rownames(counts.rna.1) <- genes
rownames(counts.rna.2) <- genes
rownames(counts.rna.3) <- genes
rownames(counts.rna.4) <- genes
rownames(counts.rna.5) <- genes
rownames(counts.rna.6) <- genes
rownames(counts.rna.7) <- genes

colnames(counts.rna.0) <- rownames(meta.cell.0)
colnames(counts.rna.1) <- rownames(meta.cell.1)
colnames(counts.rna.2) <- rownames(meta.cell.2)
colnames(counts.rna.3) <- rownames(meta.cell.3)
colnames(counts.rna.4) <- rownames(meta.cell.4)
colnames(counts.rna.5) <- rownames(meta.cell.5)
colnames(counts.rna.6) <- rownames(meta.cell.6)
colnames(counts.rna.7) <- rownames(meta.cell.7)

liger_rna <- createLiger(list(rna0 = counts.rna.0, 
                             rna1 = counts.rna.1, 
                             rna2 = counts.rna.2, 
                             rna3 = counts.rna.3, 
                             rna4 = counts.rna.4,
                             rna5 = counts.rna.5,
                             rna6 = counts.rna.6,
                             rna7 = counts.rna.7
                            ), remove.missing = FALSE)

liger_rna <- rliger::normalize(liger_rna, verbose = TRUE)

liger_rna@var.genes <- rownames(counts.rna.0)
liger_rna <- rliger::scaleNotCenter(liger_rna, remove.missing = FALSE)

num_clust <- 30
liger_rna <- rliger::optimizeALS(liger_rna, k = num_clust)
# quantile normalization
print("Writing results...")

liger_rna <- rliger::quantile_norm(liger_rna)
H0 <- liger_rna@H$rna0
H1 <- liger_rna@H$rna1
H2 <- liger_rna@H$rna2
H3 <- liger_rna@H$rna3
H4 <- liger_rna@H$rna4
H5 <- liger_rna@H$rna5
H6 <- liger_rna@H$rna6
H7 <- liger_rna@H$rna7

H.norm0 <- liger_rna@H.norm[1:dim(H0)[1],]
H.norm1 <- liger_rna@H.norm[(dim(H0)[1] + 1):(dim(H0)[1] + dim(H1)[1]),]
H.norm2 <- liger_rna@H.norm[(dim(H0)[1] + dim(H1)[1] + 1):(dim(H0)[1] + dim(H1)[1] + dim(H2)[1]),]
H.norm3 <- liger_rna@H.norm[(dim(H0)[1] + dim(H1)[1] + dim(H2)[1] + 1):(dim(H0)[1] + dim(H1)[1] + dim(H2)[1] + dim(H3)[1]),]
H.norm4 <- liger_rna@H.norm[(dim(H0)[1] + dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + 1):(dim(H0)[1] + dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1]),]
H.norm5 <- liger_rna@H.norm[(dim(H0)[1] + dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1] + 1):(dim(H0)[1] + dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1] + dim(H5)[1]),]
H.norm6 <- liger_rna@H.norm[(dim(H0)[1] + dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1] + dim(H5)[1] + 1):(dim(H0)[1] + dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1] + dim(H5)[1] + dim(H6)[1]),]
H.norm7 <- liger_rna@H.norm[(dim(H0)[1] + dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1] + dim(H5)[1] + dim(H6)[1] + 1):(dim(H0)[1] + dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1] + dim(H5)[1] + dim(H6)[1] + dim(H7)[1]),]

write.csv(H.norm0, paste0(result_dir, "H0_norm.csv"))
write.csv(H.norm1, paste0(result_dir, "H1_norm.csv"))
write.csv(H.norm2, paste0(result_dir, "H2_norm.csv"))
write.csv(H.norm3, paste0(result_dir, "H3_norm.csv"))
write.csv(H.norm4, paste0(result_dir, "H4_norm.csv"))
write.csv(H.norm5, paste0(result_dir, "H5_norm.csv"))
write.csv(H.norm6, paste0(result_dir, "H6_norm.csv"))
write.csv(H.norm7, paste0(result_dir, "H7_norm.csv"))
