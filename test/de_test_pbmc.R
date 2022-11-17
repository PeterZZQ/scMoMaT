rm(list = ls())
gc()

# install.packages('devtools')
# library(devtools)
# install_github('welch-lab/liger')
library(Seurat)
library(stringr)
library(Matrix)

setwd("/localscratch/ziqi/CFRM/test/")
# Read in the count matrix
dir <- "../data/real/ASAP-PBMC/"
results_dir <- "pbmc/scmomat/"

counts.rna.1 <- readMM(paste0(dir, 'GxC1.mtx'))
counts.rna.2 <- readMM(paste0(dir, 'GxC2.mtx'))
counts.atac.3 <- readMM(paste0(dir, 'RxC3.mtx'))
counts.atac.4 <- readMM(paste0(dir, 'RxC4.mtx'))
counts.bin.3 <- readMM(paste0(dir, 'BxC3.mtx'))
counts.bin.4 <- readMM(paste0(dir, 'BxC4.mtx'))
counts.protein.1 <- readMM(paste0(dir, 'PxC1.mtx'))
counts.protein.2 <- readMM(paste0(dir, 'PxC2.mtx'))
counts.protein.3 <- readMM(paste0(dir, 'PxC3.mtx'))
counts.protein.4 <- readMM(paste0(dir, 'PxC4.mtx'))
genes <- read.csv(paste0(dir, "genes.txt"), header = F)[[1]]
regions <- read.csv(paste0(dir, "regions.txt"), header = F)[[1]]
proteins <- read.csv(paste0(dir, "proteins.txt"), header = F)[[1]]
bins <- read.csv(paste0(dir, "bins.txt"), header = F)[[1]]
meta.cell.1 <- read.csv(paste0(dir, "meta_c1.csv"), header = T, sep = ",", row.names = 1)
meta.cell.2 <- read.csv(paste0(dir, "meta_c2.csv"), header = T, sep = ",", row.names = 1)
meta.cell.3 <- read.csv(paste0(dir, "meta_c3.csv"), header = T, sep = ",", row.names = 1)
meta.cell.4 <- read.csv(paste0(dir, "meta_c4.csv"), header = T, sep = ",", row.names = 1)
rownames(meta.cell.1) <- paste("batch_1:cell_", seq(1, dim(meta.cell.1)[1]), sep = "")
rownames(meta.cell.2) <- paste("batch_2:cell_", seq(1, dim(meta.cell.2)[1]), sep = "")
rownames(meta.cell.3) <- paste("batch_3:cell_", seq(1, dim(meta.cell.3)[1]), sep = "")
rownames(meta.cell.4) <- paste("batch_4:cell_", seq(1, dim(meta.cell.4)[1]), sep = "")

rownames(counts.rna.1) <- genes
rownames(counts.rna.2) <- genes
rownames(counts.atac.3) <- regions
rownames(counts.atac.4) <- regions
rownames(counts.bin.3) <- bins
rownames(counts.bin.4) <- bins
colnames(counts.bin.3) <- rownames(meta.cell.3)
colnames(counts.bin.4) <- rownames(meta.cell.4)
colnames(counts.rna.1) <- rownames(meta.cell.1)
colnames(counts.rna.2) <- rownames(meta.cell.2)
colnames(counts.atac.3) <- rownames(meta.cell.3)
colnames(counts.atac.4) <- rownames(meta.cell.4)
rownames(counts.protein.1) <- proteins
rownames(counts.protein.2) <- proteins
rownames(counts.protein.3) <- proteins
rownames(counts.protein.4) <- proteins
colnames(counts.protein.1) <- rownames(meta.cell.1)
colnames(counts.protein.2) <- rownames(meta.cell.2)
colnames(counts.protein.3) <- rownames(meta.cell.3)
colnames(counts.protein.4) <- rownames(meta.cell.4)


groups <- read.table(paste0(results_dir, 'leiden_30_4000_0.5.txt'), header = FALSE)
counts.rna <- cbind(counts.rna.1, counts.rna.2)
groups.rna <- groups[1:dim(counts.rna)[2],] 
meta.rna <- rbind(meta.cell.1, meta.cell.2)
meta.rna["groups"] <- groups.rna
# create seurat object
seurat.rna <- CreateSeuratObject(counts = counts.rna, 
                                 assay = "RNA", 
                                 project = "full_matrix", 
                                 meta.data = meta.rna, header = T, row.names = 1, sep = ",")

de.genes <- FindMarkers(object = seurat.rna, ident.1 = 7, ident.2 = 6, group.by = "groups")
write.csv(file = paste0(results_dir, 'de_6&7.txt'), x = de.genes)