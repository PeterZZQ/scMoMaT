rm(list = ls())
gc()

library(rliger)
library(Seurat)
library(stringr)

# tutorial see: http://htmlpreview.github.io/?https://github.com/welch-lab/liger/blob/master/vignettes/SNAREseq_walkthrough.html
setwd("/localscratch/ziqi/CFRM/test/")
# Read in the count matrix
dir <- "../data/real/MOp/"
# results_dir <- "MOp/uinmf/"
results_dir <- "MOp/uinmf_bin/"

counts.rna.1 <- readMM(paste0(dir, 'GxC1.mtx'))
counts.rna.2 <- readMM(paste0(dir, 'GxC2.mtx'))
counts.atac.1 <- readMM(paste0(dir, 'RxC1.mtx'))
counts.atac.3 <- readMM(paste0(dir, 'RxC3.mtx'))
counts.bin.1 <- readMM(paste0(dir, 'BxC1.mtx'))
counts.bin.3 <- readMM(paste0(dir, 'BxC3.mtx'))

genes <- read.csv(paste0(dir, "genes.txt"), header = F)[[1]]
regions <- read.csv(paste0(dir, "regions.txt"), header = F)[[1]]
bins <- read.csv(paste0(dir, "bins.txt"), header = F)[[1]]
meta.cell.1 <- read.csv(paste0(dir, "meta_c1.csv"), header = T, sep = ",", row.names = 1)
meta.cell.2 <- read.csv(paste0(dir, "meta_c2.csv"), header = T, sep = ",", row.names = 1)
meta.cell.3 <- read.csv(paste0(dir, "meta_c3.csv"), header = T, sep = ",", row.names = 1)

rownames(counts.rna.1) <- genes
rownames(counts.rna.2) <- genes
rownames(counts.atac.3) <- regions
rownames(counts.atac.1) <- regions
colnames(counts.rna.1) <- rownames(meta.cell.1)
colnames(counts.rna.2) <- rownames(meta.cell.2)
colnames(counts.atac.3) <- rownames(meta.cell.3)
colnames(counts.atac.1) <- rownames(meta.cell.1)
rownames(counts.bin.3) <- bins
rownames(counts.bin.1) <- bins
colnames(counts.bin.3) <- rownames(meta.cell.3)
colnames(counts.bin.1) <- rownames(meta.cell.1)


gene2region <- readMM(paste0(dir, "GxR.mtx"))
rownames(gene2region) <- genes
colnames(gene2region) <- regions

activity.matrix <- gene2region %*% counts.atac.3

# # create the liger object for the unshared region data
# liger_atac <- createLiger(list(peak1 = counts.atac.1, peak3 = counts.atac.3), remove.missing = FALSE)
# liger_atac <- normalize(liger_atac)
# norm1 <- liger_atac@norm.data$peak1
# norm2 <- liger_atac@norm.data$peak3
# norm <- cbind(norm1, norm2)

# create the liger object for the unshared bin data
liger_bin <- createLiger(list(peak1 = counts.bin.1, peak3 = counts.bin.3), remove.missing = FALSE)
liger_bin <- normalize(liger_bin)
norm1 <- liger_bin@norm.data$peak1
norm2 <- liger_bin@norm.data$peak3
norm <- cbind(norm1, norm2)

# select the top 2000 features
se = CreateSeuratObject(norm)
vars_2000 <- FindVariableFeatures(se, selection.method = "vst", nfeatures = 2000)
top2000 <- head(VariableFeatures(vars_2000),2000)
# rename the - into _
top2000 <-lapply(top2000, function(x){
  x <- strsplit(x, split = "-")[[1]]
  x <- paste0(x[1], "_", x[2], "_", x[3])
  return(x)})
top2000 <- unlist(top2000)
top2000_feats <-  norm[top2000,]   

# liger_atac <- selectGenes(liger_atac)
# liger_atac@var.genes <- top2000
# liger_atac <- scaleNotCenter(liger_atac)
# unshared_atac1 = liger_atac@scale.data$peak1
# unshared_atac3 = liger_atac@scale.data$peak3

liger_bin <- selectGenes(liger_bin)
liger_bin@var.genes <- top2000
liger_bin <- scaleNotCenter(liger_bin)
unshared_atac1 = liger_bin@scale.data$peak1
unshared_atac3 = liger_bin@scale.data$peak3

# create the liger object and normalize the shared data
ifnb_liger <- createLiger(list(rna1 = counts.rna.1, rna2 = counts.rna.2, rna3 = activity.matrix), remove.missing = FALSE)
ifnb_liger <- normalize(ifnb_liger)
# for protein we don't select has there are only 216 proteins
ifnb_liger <- selectGenes(ifnb_liger)
# scale the data
ifnb_liger <- scaleNotCenter(ifnb_liger)

# Add the unshared features that have been properly selected, such that they are added as a genes by cells matrix.
ifnb_liger@var.unshared.features[[1]] = colnames(unshared_atac1)
ifnb_liger@scale.unshared.data[[1]] = t(unshared_atac1)
ifnb_liger@var.unshared.features[[3]] = colnames(unshared_atac3)
ifnb_liger@scale.unshared.data[[3]] = t(unshared_atac3)
# Joint matrix factorization
ifnb_liger <- optimizeALS(ifnb_liger, k=30, use.unshared = TRUE, max_iters =30,thresh=1e-10)


ifnb_liger <- quantile_norm(ifnb_liger)
H1 <- ifnb_liger@H$rna1
H2 <- ifnb_liger@H$rna2
H3 <- ifnb_liger@H$rna3

H.norm1 <- ifnb_liger@H.norm[1:dim(H1)[1],]
H.norm2 <- ifnb_liger@H.norm[(dim(H1)[1] + 1):(dim(H1)[1] + dim(H2)[1]),]
H.norm3 <- ifnb_liger@H.norm[(dim(H1)[1] + dim(H2)[1] + 1):(dim(H1)[1] + dim(H2)[1] + dim(H3)[1]),]

write.csv(H1, paste0(results_dir, "H1.csv"))
write.csv(H2, paste0(results_dir, "H2.csv"))
write.csv(H3, paste0(results_dir, "H3.csv"))

write.csv(H.norm1, paste0(results_dir, "H1_norm.csv"))
write.csv(H.norm2, paste0(results_dir, "H2_norm.csv"))
write.csv(H.norm3, paste0(results_dir, "H3_norm.csv"))

# plot
ifnb_liger <- louvainCluster(ifnb_liger)
ifnb_liger <- runUMAP(ifnb_liger)
umap_plots <-plotByDatasetAndCluster(ifnb_liger, axis.labels = c("UMAP1","UMAP2"), return.plots = TRUE)
umap_plots[[2]]

