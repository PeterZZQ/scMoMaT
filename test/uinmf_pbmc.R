rm(list = ls())
gc()

# install.packages('devtools')
# library(devtools)
# install_github('welch-lab/liger')
library(rliger)
library(Seurat)
library(stringr)

setwd("/localscratch/ziqi/CFRM/test/")
# Read in the count matrix
dir <- "../data/real/ASAP-PBMC/"
# results_dir <- "pbmc/uinmf/"
results_dir <- "pbmc/uinmf_bin/"

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
# create the liger object for the unshared data
# liger_atac <- createLiger(list(peak1 = counts.atac.3, peak2 = counts.atac.4), remove.missing = FALSE)
# liger_atac <- normalize(liger_atac)
# norm1 <- liger_atac@norm.data$peak1
# norm2 <- liger_atac@norm.data$peak2
# norm <- cbind(norm1, norm2)

liger_bin <- createLiger(list(peak1 = counts.bin.3, peak2 = counts.bin.4), remove.missing = FALSE)
liger_bin <- normalize(liger_bin)
norm1 <- liger_bin@norm.data$peak1
norm2 <- liger_bin@norm.data$peak2
norm <- cbind(norm1, norm2)

# select the top 2000 features
se = CreateSeuratObject(norm)
vars_2000 <- FindVariableFeatures(se, selection.method = "vst", nfeatures = 2000)
top2000 <- head(VariableFeatures(vars_2000),2000)
# rename the - into _
top2000 <- lapply(top2000, function(x){
  x <- strsplit(x, split = "-")[[1]]
  x <- paste0(x[1], "_", x[2], "_", x[3])
  return(x)})
top2000 <- unlist(top2000)
top2000_feats <-  norm[top2000,]   

# liger_atac <- selectGenes(liger_atac)
# liger_atac@var.genes <- top2000
# liger_atac <- scaleNotCenter(liger_atac)
# unshared_atac1 = liger_atac@scale.data$peak1
# unshared_atac2 = liger_atac@scale.data$peak2
liger_bin <- selectGenes(liger_bin)
liger_bin@var.genes <- top2000
liger_bin <- scaleNotCenter(liger_bin)
unshared_atac1 = liger_bin@scale.data$peak1
unshared_atac2 = liger_bin@scale.data$peak2


liger_rna <- createLiger(list(rna1 = counts.rna.1, rna2 = counts.rna.2), remove.missing = FALSE)
liger_rna <- normalize(liger_rna)
norm1 <- liger_rna@norm.data$rna1
norm2 <- liger_rna@norm.data$rna2
norm <- cbind(norm1, norm2)

# select the top 2000 features
se = CreateSeuratObject(norm)
vars_2000 <- FindVariableFeatures(se, selection.method = "vst", nfeatures = 2000)
top2000 <- head(VariableFeatures(vars_2000),2000)
top2000_feats <-  norm[top2000,]   

liger_rna <- selectGenes(liger_rna)
liger_rna@var.genes <- top2000
liger_rna <- scaleNotCenter(liger_rna)
unshared_rna1 = liger_rna@scale.data$rna1
unshared_rna2 = liger_rna@scale.data$rna2


# create the liger object and normalize the shared data
ifnb_liger <- createLiger(list(protein1 = counts.protein.1, protein2 = counts.protein.2, protein3 = counts.protein.3, protein4 = counts.protein.4), remove.missing = FALSE)
ifnb_liger <- normalize(ifnb_liger)
# for protein we don't select has there are only 216 proteins
ifnb_liger <- selectGenes(ifnb_liger)
ifnb_liger <- scaleNotCenter(ifnb_liger)


# scale the data
ifnb_liger <- scaleNotCenter(ifnb_liger)

# Add the unshared features that have been properly selected, such that they are added as a genes by cells matrix.
ifnb_liger@var.unshared.features[[1]] = colnames(unshared_rna1)
ifnb_liger@scale.unshared.data[[1]] = t(unshared_rna1)
ifnb_liger@var.unshared.features[[2]] = colnames(unshared_rna2)
ifnb_liger@scale.unshared.data[[2]] = t(unshared_rna2)
ifnb_liger@var.unshared.features[[3]] = colnames(unshared_atac1)
ifnb_liger@scale.unshared.data[[3]] = t(unshared_atac1)
ifnb_liger@var.unshared.features[[4]] = colnames(unshared_atac2)
ifnb_liger@scale.unshared.data[[4]] = t(unshared_atac2)

# Joint matrix factorization
ifnb_liger <- optimizeALS(ifnb_liger, k=30, use.unshared = TRUE, max_iters =30,thresh=1e-10)


ifnb_liger <- quantile_norm(ifnb_liger)
H1 <- ifnb_liger@H$protein1
H2 <- ifnb_liger@H$protein2
H3 <- ifnb_liger@H$protein3
H4 <- ifnb_liger@H$protein4

H.norm1 <- ifnb_liger@H.norm[1:dim(H1)[1],]
H.norm2 <- ifnb_liger@H.norm[(dim(H1)[1] + 1):(dim(H1)[1] + dim(H2)[1]),]
H.norm3 <- ifnb_liger@H.norm[(dim(H1)[1] + dim(H2)[1] + 1):(dim(H1)[1] + dim(H2)[1] + dim(H3)[1]),]
H.norm4 <- ifnb_liger@H.norm[(dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + 1):(dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1]),]

write.csv(H1, paste0(results_dir, "H1.csv"))
write.csv(H2, paste0(results_dir, "H2.csv"))
write.csv(H3, paste0(results_dir, "H3.csv"))
write.csv(H4, paste0(results_dir, "H4.csv"))

write.csv(H.norm1, paste0(results_dir, "H1_norm.csv"))
write.csv(H.norm2, paste0(results_dir, "H2_norm.csv"))
write.csv(H.norm3, paste0(results_dir, "H3_norm.csv"))
write.csv(H.norm4, paste0(results_dir, "H4_norm.csv"))

# plot
ifnb_liger <- louvainCluster(ifnb_liger)
ifnb_liger <- runUMAP(ifnb_liger)
umap_plots <-plotByDatasetAndCluster(ifnb_liger, axis.labels = c("UMAP1","UMAP2"), return.plots = TRUE)
umap_plots[[2]]

