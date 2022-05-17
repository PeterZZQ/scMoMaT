rm(list =ls())
gc()
library(Signac)
library(Seurat)
library(GenomeInfoDb)
library(ggplot2)
library(patchwork)
library(Matrix)
setwd("/localscratch/ziqi/CFRM/test/")

# Read in the raw data
path <- "../data/real/diag/mouse_brain_cortex/"
result_path <- "mbc/liger/"
counts.atac <- readMM(paste0(path, "RxC2.mtx"))
rownames(counts.atac) <- read.table(paste0(path, "regions.txt"), header = F, sep = ",")[[1]]
colnames(counts.atac) <- rownames(read.table(paste0(path, "meta_c2.csv"), header = T, row.names = 1, sep = ","))
counts.rna <- readMM(paste0(path, "GxC1.mtx"))
rownames(counts.rna) <- read.table(paste0(path, "genes.txt"), header = F, sep = ",")[[1]]
colnames(counts.rna) <- rownames(read.table(paste0(path, "meta_c1.csv"), header = T, row.names = 1, sep = ","))
gene2region <- readMM(paste0(path, "GxR.mtx"))
rownames(gene2region) <- rownames(counts.rna)
colnames(gene2region) <- rownames(counts.atac)

bins <- read.csv(paste0(path, "bins.txt"), header = F)[[1]]
counts.bin <- readMM(paste0(path, 'BxC2.mtx'))
rownames(counts.bin) <- bins
colnames(counts.bin) <- colnames(counts.atac)

################################################
#
# Using LIGER
#
################################################
activity.matrix <- gene2region %*% counts.atac

ifnb_liger <- rliger::createLiger(list(rna1 = counts.rna, atac2 = activity.matrix), remove.missing = FALSE)
ifnb_liger <- rliger::normalize(ifnb_liger, verbose = TRUE)
ifnb_liger <- rliger::selectGenes(ifnb_liger, datasets.use = 2)
# select all the genes
# ifnb_liger@var.genes <- paste("Gene_", seq(1, dim(region2gene)[2]), sep = "")
ifnb_liger <- rliger::scaleNotCenter(ifnb_liger, remove.missing = FALSE)

num_clust <- 20
ifnb_liger <- rliger::optimizeALS(ifnb_liger, k = num_clust)
# quantile normalization
print("Writing results...")

ifnb_liger <- rliger::quantile_norm(ifnb_liger)
H1 <- ifnb_liger@H$rna1
H2 <- ifnb_liger@H$atac2

H.norm1 <- ifnb_liger@H.norm[1:dim(H1)[1],]
H.norm2 <- ifnb_liger@H.norm[(dim(H1)[1] + 1):(dim(H1)[1] + dim(H2)[1]),]
write.csv(H.norm1, paste0(result_path, "liger_H1.csv"))
write.csv(H.norm2, paste0(result_path, "liger_H2.csv"))


################################################
#
# Using UINMF
#
################################################
result_path <- "spleen/uinmf/"
liger <- createLiger(list(peaks = counts.atac))
liger <- normalize(liger)
norm <- liger@norm.data$peaks

# filtering the unshared features
se = CreateSeuratObject(norm)
vars_2000 <- FindVariableFeatures(se, selection.method = "vst", nfeatures = 2000)
top2000 <- head(VariableFeatures(vars_2000), 2000)
# rename the - into _
top2000 <-lapply(top2000, function(x){
  x <- strsplit(x, split = "-")[[1]]
  x <- paste0(x[1], "_", x[2], "_", x[3])
  return(x)})
top2000 <- unlist(top2000)
top2000_feats <-  norm[top2000,]
liger <- selectGenes(liger)
liger@var.genes <- top2000
liger <- scaleNotCenter(liger)
unshared_feats = liger@scale.data$peaks

# filtering the shared features
activity.matrix <- gene2region %*% counts.atac
liger <- rliger::createLiger(list(rna1 = counts.rna, atac2 = activity.matrix), remove.missing = FALSE)
liger <- rliger::normalize(liger, verbose = TRUE)
liger <- rliger::selectGenes(liger, datasets.use = 1, unshared = TRUE,  unshared.datasets = list(2), unshared.thresh= 0.2)
liger <- scaleNotCenter(liger)

# add in the unshared features
peak_names <- rownames(unshared_feats)
liger@var.unshared.features[[2]] = peak_names
liger@scale.unshared.data[[2]] = t(unshared_feats)

# Joint matrix factorization
liger <- optimizeALS(liger, k=30, use.unshared = TRUE, max_iters =30,thresh=1e-10)

# quantile normalization
liger <- quantile_norm(liger)

# save results
H1 <- liger@H$rna1
H2 <- liger@H$atac2
write.csv(H1, paste0(result_path, "liger_c1.csv"))
write.csv(H2, paste0(result_path, "liger_c2.csv"))
H.norm1 <- liger@H.norm[1:dim(H1)[1],]
H.norm2 <- liger@H.norm[(dim(H1)[1] + 1):(dim(H1)[1] + dim(H2)[1]),]
write.csv(H.norm1, paste0(result_path, "liger_c1_norm.csv"))
write.csv(H.norm2, paste0(result_path, "liger_c2_norm.csv"))

# plot
liger <- louvainCluster(liger)
liger <- runUMAP(liger)
umap_plots <-plotByDatasetAndCluster(liger, axis.labels = c("UMAP1","UMAP2"), return.plots = TRUE)
umap_plots[[2]]


################################################
#
# Using UINMF with bins
#
################################################
result_path <- "spleen/uinmf_bin/"
liger <- createLiger(list(peaks = counts.bin))
liger <- normalize(liger)
norm <- liger@norm.data$peaks

# filtering the unshared features
se = CreateSeuratObject(norm)
vars_2000 <- FindVariableFeatures(se, selection.method = "vst", nfeatures = 2000)
top2000 <- head(VariableFeatures(vars_2000), 2000)
# rename the - into _
top2000 <-lapply(top2000, function(x){
  x <- strsplit(x, split = "-")[[1]]
  x <- paste0(x[1], "_", x[2], "_", x[3])
  return(x)})
top2000 <- unlist(top2000)
top2000_feats <-  norm[top2000,]
liger <- selectGenes(liger)
liger@var.genes <- top2000
liger <- scaleNotCenter(liger)
unshared_feats = liger@scale.data$peaks

# filtering the shared features
activity.matrix <- gene2region %*% counts.atac
liger <- rliger::createLiger(list(rna1 = counts.rna, atac2 = activity.matrix), remove.missing = FALSE)
liger <- rliger::normalize(liger, verbose = TRUE)
liger <- rliger::selectGenes(liger, datasets.use = 1, unshared = TRUE,  unshared.datasets = list(2), unshared.thresh= 0.2)
liger <- scaleNotCenter(liger)

# add in the unshared features
peak_names <- rownames(unshared_feats)
liger@var.unshared.features[[2]] = peak_names
liger@scale.unshared.data[[2]] = t(unshared_feats)

# Joint matrix factorization
liger <- optimizeALS(liger, k=30, use.unshared = TRUE, max_iters =30,thresh=1e-10)

# quantile normalization
liger <- quantile_norm(liger)

# save results
H1 <- liger@H$rna1
H2 <- liger@H$atac2
write.csv(H1, paste0(result_path, "liger_c1.csv"))
write.csv(H2, paste0(result_path, "liger_c2.csv"))
H.norm1 <- liger@H.norm[1:dim(H1)[1],]
H.norm2 <- liger@H.norm[(dim(H1)[1] + 1):(dim(H1)[1] + dim(H2)[1]),]
write.csv(H.norm1, paste0(result_path, "liger_c1_norm.csv"))
write.csv(H.norm2, paste0(result_path, "liger_c2_norm.csv"))

# plot
liger <- louvainCluster(liger)
liger <- runUMAP(liger)
umap_plots <-plotByDatasetAndCluster(liger, axis.labels = c("UMAP1","UMAP2"), return.plots = TRUE)
umap_plots[[2]]


