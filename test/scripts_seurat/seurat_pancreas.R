rm(list =ls())
gc()
library(Signac)
library(Seurat)
library(GenomeInfoDb)
library(ggplot2)
library(patchwork)
library(Matrix)

setwd("/localscratch/ziqi/scMoMaT/test/scripts_seurat/")
# Read in the count matrix
dir <- "../../data/real/hori/Pancreas/"
result_path <- "../pancreas/Seurat/"

counts.rna.0 <- readMM(paste0(dir, 'GxC0.mtx'))
counts.rna.1 <- readMM(paste0(dir, 'GxC1.mtx'))
counts.rna.2 <- readMM(paste0(dir, 'GxC2.mtx'))
counts.rna.3 <- readMM(paste0(dir, 'GxC3.mtx'))
counts.rna.4 <- readMM(paste0(dir, 'GxC4.mtx'))
counts.rna.5 <- readMM(paste0(dir, 'GxC5.mtx'))
counts.rna.6 <- readMM(paste0(dir, 'GxC6.mtx'))
counts.rna.7 <- readMM(paste0(dir, 'GxC7.mtx'))

genes <- read.csv(paste0(dir, "gene.csv"), header = F)[[1]]
meta.cell.0 <- read.csv(paste0(dir, "meta_c0.csv"), header = T, sep = ",", row.names = 1)
meta.cell.1 <- read.csv(paste0(dir, "meta_c1.csv"), header = T, sep = ",", row.names = 1)
meta.cell.2 <- read.csv(paste0(dir, "meta_c2.csv"), header = T, sep = ",", row.names = 1)
meta.cell.3 <- read.csv(paste0(dir, "meta_c3.csv"), header = T, sep = ",", row.names = 1)
meta.cell.4 <- read.csv(paste0(dir, "meta_c4.csv"), header = T, sep = ",", row.names = 1)
meta.cell.5 <- read.csv(paste0(dir, "meta_c5.csv"), header = T, sep = ",", row.names = 1)
meta.cell.6 <- read.csv(paste0(dir, "meta_c6.csv"), header = T, sep = ",", row.names = 1)
meta.cell.7 <- read.csv(paste0(dir, "meta_c7.csv"), header = T, sep = ",", row.names = 1)

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


# create seurat object-RNA
seurat.rna.0 <- CreateSeuratObject(counts = counts.rna.0, assay = "RNA", project = "full_matrix", meta.data = meta.cell.0, header = T, row.names = 1, sep = ",")

seurat.rna.1 <- CreateSeuratObject(counts = counts.rna.1, assay = "RNA", project = "full_matrix", meta.data = meta.cell.1, header = T, row.names = 1, sep = ",")

seurat.rna.2 <- CreateSeuratObject(counts = counts.rna.2, assay = "RNA", project = "full_matrix", meta.data = meta.cell.2, header = T, row.names = 1, sep = ",")

seurat.rna.3 <- CreateSeuratObject(counts = counts.rna.3, assay = "RNA", project = "full_matrix", meta.data = meta.cell.3, header = T, row.names = 1, sep = ",")

seurat.rna.4 <- CreateSeuratObject(counts = counts.rna.4, assay = "RNA", project = "full_matrix", meta.data = meta.cell.4, header = T, row.names = 1, sep = ",")

seurat.rna.5 <- CreateSeuratObject(counts = counts.rna.5, assay = "RNA", project = "full_matrix", meta.data = meta.cell.5, header = T, row.names = 1, sep = ",")

seurat.rna.6 <- CreateSeuratObject(counts = counts.rna.6, assay = "RNA", project = "full_matrix", meta.data = meta.cell.6, header = T, row.names = 1, sep = ",")

seurat.rna.7 <- CreateSeuratObject(counts = counts.rna.7, assay = "RNA", project = "full_matrix", meta.data = meta.cell.7, header = T, row.names = 1, sep = ",")

seurat.rnas <- list(seurat.rna.0, seurat.rna.1, seurat.rna.2, seurat.rna.3, seurat.rna.4, seurat.rna.5, seurat.rna.6, seurat.rna.7)
seurat.rnas <- lapply(X = seurat.rnas, FUN = function(x) {
    x <- NormalizeData(x)
    x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
})

# select features that are repeatedly variable across datasets for integration
features <- SelectIntegrationFeatures(object.list = seurat.rnas)
# find integration anchors
seurat.anchors <- FindIntegrationAnchors(object.list = seurat.rnas, anchor.features = features)
# this command creates an 'integrated' data assay
seurat.combined <- IntegrateData(anchorset = seurat.anchors)

# specify that we will perform downstream analysis on the corrected data note that the
# original unmodified data still resides in the 'RNA' assay
DefaultAssay(seurat.combined) <- "integrated"

# Run the standard workflow for visualization and clustering
seurat.combined <- ScaleData(seurat.combined, verbose = FALSE)
seurat.combined <- RunPCA(seurat.combined, npcs = 30, verbose = FALSE)
seurat.combined <- RunUMAP(seurat.combined, reduction = "pca", dims = 1:30)
seurat.combined <- FindNeighbors(seurat.combined, reduction = "pca", dims = 1:30)
seurat.combined <- FindClusters(seurat.combined, resolution = 0.5)

pca_embedding <- seurat.combined@reductions$pca@cell.embeddings
umap_embedding <- seurat.combined@reductions$umap@cell.embeddings

pca_embedding0 <- pca_embedding[1:dim(counts.rna.0)[2],]
pca_embedding1 <- pca_embedding[(dim(counts.rna.0)[2] + 1):(dim(counts.rna.0)[2] + dim(counts.rna.1)[2]),]
pca_embedding2 <- pca_embedding[(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + 1):(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2]),]
pca_embedding3 <- pca_embedding[(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + 1):(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2]),]
pca_embedding4 <- pca_embedding[(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + 1):(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2]),]
pca_embedding5 <- pca_embedding[(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + 1):(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + dim(counts.rna.5)[2]),]
pca_embedding6 <- pca_embedding[(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + dim(counts.rna.5)[2] + 1):(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + dim(counts.rna.5)[2] + dim(counts.rna.6)[2]),]
pca_embedding7 <- pca_embedding[(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + dim(counts.rna.5)[2] + dim(counts.rna.6)[2] + 1):(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + dim(counts.rna.5)[2] + dim(counts.rna.6)[2] + dim(counts.rna.7)[2]),]

umap_embedding0 <- umap_embedding[1:dim(counts.rna.0)[2],]
umap_embedding1 <- umap_embedding[(dim(counts.rna.0)[2] + 1):(dim(counts.rna.0)[2] + dim(counts.rna.1)[2]),]
umap_embedding2 <- umap_embedding[(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + 1):(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2]),]
umap_embedding3 <- umap_embedding[(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + 1):(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2]),]
umap_embedding4 <- umap_embedding[(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + 1):(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2]),]
umap_embedding5 <- umap_embedding[(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + 1):(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + dim(counts.rna.5)[2]),]
umap_embedding6 <- umap_embedding[(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + dim(counts.rna.5)[2] + 1):(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + dim(counts.rna.5)[2] + dim(counts.rna.6)[2]),]
umap_embedding7 <- umap_embedding[(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + dim(counts.rna.5)[2] + dim(counts.rna.6)[2] + 1):(dim(counts.rna.0)[2] + dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + dim(counts.rna.5)[2] + dim(counts.rna.6)[2] + dim(counts.rna.7)[2]),]

write.table(pca_embedding, file = paste0(result_path, "seurat_pca.txt"), sep = "\t")
write.table(umap_embedding, file = paste0(result_path, "seurat_umap.txt"), sep = "\t")

write.table(pca_embedding0, file = paste0(result_path, "seurat_pca0.txt"), sep = "\t")
write.table(pca_embedding1, file = paste0(result_path, "seurat_pca1.txt"), sep = "\t")
write.table(pca_embedding2, file = paste0(result_path, "seurat_pca2.txt"), sep = "\t")
write.table(pca_embedding3, file = paste0(result_path, "seurat_pca3.txt"), sep = "\t")
write.table(pca_embedding4, file = paste0(result_path, "seurat_pca4.txt"), sep = "\t")
write.table(pca_embedding5, file = paste0(result_path, "seurat_pca5.txt"), sep = "\t")
write.table(pca_embedding6, file = paste0(result_path, "seurat_pca6.txt"), sep = "\t")
write.table(pca_embedding7, file = paste0(result_path, "seurat_pca7.txt"), sep = "\t")

write.table(umap_embedding0, file = paste0(result_path, "seurat_umap0.txt"), sep = "\t")
write.table(umap_embedding1, file = paste0(result_path, "seurat_umap1.txt"), sep = "\t")
write.table(umap_embedding2, file = paste0(result_path, "seurat_umap2.txt"), sep = "\t")
write.table(umap_embedding3, file = paste0(result_path, "seurat_umap3.txt"), sep = "\t")
write.table(umap_embedding4, file = paste0(result_path, "seurat_umap4.txt"), sep = "\t")
write.table(umap_embedding5, file = paste0(result_path, "seurat_umap5.txt"), sep = "\t")
write.table(umap_embedding6, file = paste0(result_path, "seurat_umap6.txt"), sep = "\t")
write.table(umap_embedding7, file = paste0(result_path, "seurat_umap7.txt"), sep = "\t")




