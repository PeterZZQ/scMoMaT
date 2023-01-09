# # genome assembly and gene annotation packages
# # Mouse mm10 
# BiocManager::install(c('BSgenome.Mmusculus.UCSC.mm10', 'EnsDb.Mmusculus.v79'))
# # Human hg19
# BiocManager::install(c('BSgenome.Hsapiens.UCSC.hg19', 'EnsDb.Hsapiens.v75'))
# # Human hg38
# BiocManager::install(c('BSgenome.Hsapiens.UCSC.hg38', 'EnsDb.Hsapiens.v86'))
# install seurat v3.2
# need to use old spatstat
# install.packages('https://cran.r-project.org/src/contrib/Archive/spatstat/spatstat_1.64-1.tar.gz', repos=NULL,type="source")
# remotes::install_version("Seurat", version = "3.2")

rm(list =ls())
gc()
library(Signac)
library(Seurat)
library(GenomeInfoDb)
library(ggplot2)
library(patchwork)
library(Matrix)
setwd("/localscratch/ziqi/scMoMaT/test/scripts_seurat/")


################################################
#
# Using Raw data
#
################################################


# Read in the raw data
# do not remove cell types
path <- "../../data/real/diag/spleen/"
result_path <- "../spleen/seurat/"

# remove some cell types
path <- "../../data/real/diag/spleen/remove_celltype/"
result_path <- "../spleen/remove_celltype/seurat/"
counts.atac <- readMM(paste0(path, "RxC2.mtx"))
rownames(counts.atac) <- read.table(paste0(path, "regions.txt"), header = F, sep = ",")[[1]]
colnames(counts.atac) <- rownames(read.csv(paste0(path, "meta_c2.csv"), header = T, row.names = 1, sep = ","))
meta.atac <- read.csv(paste0(path, "meta_c2.csv"), header = T, row.names = 1, sep = ",")

counts.rna <- readMM(paste0(path, "GxC1.mtx"))
rownames(counts.rna) <- read.table(paste0(path, "genes.txt"), header = F, sep = ",")[[1]]
colnames(counts.rna) <- rownames(read.csv(paste0(path, "meta_c1.csv"), header = T, row.names = 1, sep = ","))
gene2region <- readMM(paste0(path, "GxR.mtx"))
rownames(gene2region) <- rownames(counts.rna)
colnames(gene2region) <- rownames(counts.atac)
meta.rna <- read.csv(paste0(path, "meta_c1.csv"), header = T, row.names = 1, sep = ",")


# For mouse brain cortex:
atac_assay <- CreateChromatinAssay(
  counts = counts.atac,
  sep = c("_", "_"),
  genome = "mm10"
)

seurat.atac <- CreateSeuratObject(
  counts = atac_assay,
  assay = 'peaks',
  project = 'ATAC',
  meta.data = meta.atac, header = T, row.names = 1, sep = ",")


# Not using the annotation as we use the self-calculated gene activity matrix
# # extract gene annotations from EnsDb
# # For bmmc dataset:
# annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v75)
# # For mouse brain cortex dataset:
# annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Mmusculus.v79)
# # change to UCSC style since the data was mapped to hg19
# seqlevelsStyle(annotations) <- 'UCSC'
# # add the gene information to the object
# Annotation(seurat.atac) <- annotations


# create seurat object-RNA
seurat.rna <- CreateSeuratObject(counts = counts.rna, 
                                 assay = "RNA", 
                                 project = "full_matrix", 
                                 meta.data = meta.rna, header = T, row.names = 1, sep = ",")

DefaultAssay(seurat.rna) <- "RNA"
# pre-processing
seurat.rna <- NormalizeData(seurat.rna)
seurat.rna <- FindVariableFeatures(seurat.rna)
seurat.rna <- ScaleData(seurat.rna)
seurat.rna <- RunPCA(seurat.rna)
seurat.rna <- RunUMAP(seurat.rna, dims = 1:30)

# We exclude the first dimension as this is typically correlated with sequencing depth
seurat.atac <- RunTFIDF(seurat.atac)
seurat.atac <- FindTopFeatures(seurat.atac, min.cutoff = "q0")
seurat.atac <- RunSVD(seurat.atac)
seurat.atac <- RunUMAP(seurat.atac, reduction = "lsi", dims = 2:30, reduction.name = "umap.atac", reduction.key = "atacUMAP_")

# Plot the data
p1 <- DimPlot(seurat.rna, group.by = "cell_type", label = TRUE) + NoLegend() + ggtitle("RNA")
p2 <- DimPlot(seurat.atac, group.by = "cell_type", label = TRUE) + NoLegend() + ggtitle("ATAC")
p1 + p2

# Calculating the gene activity matrix from scATAC-Seq data
# compute gene activities
# gene.activities <- GeneActivity(seurat.atac, assay = "peaks", features = VariableFeatures(seurat.rna))
# # add the gene activity matrix to the Seurat object as a new assay
# seurat.atac[['RNA']] <- CreateAssayObject(counts = gene.activities)
seurat.atac[['RNA']] <- CreateAssayObject(counts = gene2region %*% counts.atac)
seurat.atac <- NormalizeData(
  object = seurat.atac,
  assay = 'RNA',
  normalization.method = 'LogNormalize',
  scale.factor = median(seurat.atac$nCount_RNA)
)
DefaultAssay(seurat.atac) <- 'RNA'



transfer.anchors <- FindTransferAnchors(
  reference = seurat.rna,
  query = seurat.atac,
  # features = VariableFeatures(object = seurat.rna),
  reduction = 'cca',
  dims = 1:40
)

# Label transfer
predicted.labels <- TransferData(
  anchorset = transfer.anchors,
  refdata = seurat.rna@meta.data$cell_type,
  weight.reduction = seurat.atac[['lsi']],
  dims = 2:30
)
seurat.atac <- AddMetaData(object = seurat.atac, metadata = predicted.labels)
# Plot the result
plot1 <- DimPlot(seurat.atac, group.by = 'cell_type', label = TRUE, repel = TRUE) + NoLegend() + ggtitle('Ground truth annotation')
plot2 <- DimPlot(seurat.atac, group.by = 'predicted.id', label = TRUE, repel = TRUE) + NoLegend() + ggtitle('Predicted annotation')
plot1 + plot2


# coembedding
# note that we restrict the imputation to variable genes from scRNA-seq, but could impute the
# full transcriptome if we wanted to
genes.use <- VariableFeatures(seurat.rna)
refdata <- GetAssayData(seurat.rna, assay = "RNA", slot = "data")[genes.use, ]

# refdata (input) contains a scRNA-seq expression matrix for the scRNA-seq cells.  imputation
# (output) will contain an imputed scRNA-seq matrix for each of the ATAC cells
imputation <- TransferData(anchorset = transfer.anchors, refdata = refdata, weight.reduction = seurat.atac[["lsi"]],
                           dims = 2:30)
seurat.atac[["RNA"]] <- imputation

coembed <- merge(x = seurat.rna, y = seurat.atac)

# Finally, we run PCA and UMAP on this combined object, to visualize the co-embedding of both
# datasets
coembed <- ScaleData(coembed, features = genes.use, do.scale = FALSE)
coembed <- RunPCA(coembed, features = genes.use, verbose = FALSE)
coembed <- RunUMAP(coembed, dims = 1:30)

# Plot the dataset
DimPlot(coembed, group.by = c("cell_type"))

write.table(coembed@reductions$pca@cell.embeddings[1:dim(seurat.rna)[2], ], file = paste0(result_path, "seurat_pca_c1.txt"), sep = "\t")
write.table(coembed@reductions$pca@cell.embeddings[(dim(seurat.rna)[2]+1):(dim(seurat.rna)[2] + dim(seurat.atac)[2]), ], file = paste0(result_path, "seurat_pca_c2.txt"), sep = "\t")
write.table(coembed@reductions$umap@cell.embeddings[1:dim(seurat.rna)[2], ], file = paste0(result_path, "seurat_umap_c1.txt"), sep = "\t")
write.table(coembed@reductions$umap@cell.embeddings[(dim(seurat.rna)[2]+1):(dim(seurat.rna)[2] + dim(seurat.atac)[2]), ], file = paste0(result_path, "seurat_umap_c2.txt"), sep = "\t")
