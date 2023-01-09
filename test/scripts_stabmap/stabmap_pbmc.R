rm(list = ls())
gc()

library(StabMap)
library(SingleCellMultiModal)
library(scran)
library(MultiAssayExperiment)
library(scater)
# useful tutorial to create MultiAssayExperiment class: https://www.bioconductor.org/packages/devel/bioc/vignettes/MultiAssayExperiment/inst/doc/MultiAssayExperiment.html
setwd("/localscratch/ziqi/scMoMaT/test/scripts_stabmap/")

data_dir <- "../../data/real/ASAP-PBMC/"
results_dir <- "../pbmc/stabmap/"
# mae <- scMultiome("pbmc_10x", mode = "*", dry.run = FALSE, format = "MTX")
# upsetSamples(mae)
# # normalize
# sce.rna <- experiments(mae)[["rna"]]
# # Normalisation
# sce.rna <- logNormCounts(sce.rna)
# # Feature selection
# decomp <- modelGeneVar(sce.rna)
# hvgs <- rownames(decomp)[decomp$mean>0.01 & decomp$p.value <= 0.05]
# length(hvgs)
# sce.rna <- sce.rna[hvgs,]
# 
# sce.atac <- experiments(mae)[["atac"]]
# # Normalise
# sce.atac <- logNormCounts(sce.atac)
# # Feature selection using highly variable peaks
# # And adding matching peaks to genes
# decomp <- modelGeneVar(sce.atac)
# hvgs <- rownames(decomp)[decomp$mean>0.25
#                          & decomp$p.value <= 0.05]
# length(hvgs)
# sce.atac <- sce.atac[hvgs,]
# # combine the log-normalized counts of rna and atac
# logcounts_all = rbind(logcounts(sce.rna), logcounts(sce.atac))
# dim(logcounts_all)
# assayType = ifelse(rownames(logcounts_all) %in% rownames(sce.rna), "rna", "atac")
# 
# # for each barcode/cell, give it RNA or Multiome origin
# dataType = setNames(sample(c("RNA", "Multiome"), ncol(logcounts_all), prob = c(0.5,0.5), replace = TRUE), colnames(logcounts_all))

# read in the dataset
GxC1 <- readMM(paste0(data_dir, "GxC1.mtx")) 
GxC2 <- readMM(paste0(data_dir, "GxC2.mtx")) 
PxC1 <- readMM(paste0(data_dir, "PxC1.mtx")) 
PxC2 <- readMM(paste0(data_dir, "PxC2.mtx")) 
PxC3 <- readMM(paste0(data_dir, "PxC3.mtx")) 
PxC4 <- readMM(paste0(data_dir, "PxC4.mtx")) 
RxC3 <- readMM(paste0(data_dir, "RxC3.mtx")) 
RxC4 <- readMM(paste0(data_dir, "RxC4.mtx"))
# features
genes <- read.csv(paste0(data_dir, "genes.txt") , header = FALSE)
regions <- read.csv(paste0(data_dir, "regions.txt"), header = FALSE)
proteins <- read.csv(paste0(data_dir, "proteins.txt"), header = FALSE)
# meta_data
meta_1 <- read.csv(paste0(data_dir, "meta_c1.csv"), row.names = 1)
meta_2 <- read.csv(paste0(data_dir, "meta_c2.csv"), row.names = 1)
meta_3 <- read.csv(paste0(data_dir, "meta_c3.csv"), row.names = 1)
meta_4 <- read.csv(paste0(data_dir, "meta_c4.csv"), row.names = 1)
rownames(meta_1) <- paste("batch_1:", rownames(meta_1), sep = "")
rownames(meta_2) <- paste("batch_2:", rownames(meta_2), sep = "")
rownames(meta_3) <- paste("batch_3:", rownames(meta_3), sep = "")
rownames(meta_4) <- paste("batch_4:", rownames(meta_4), sep = "")

rownames(GxC1) <- genes[[1]]
rownames(GxC2) <- genes[[1]]
rownames(RxC3) <- regions[[1]]
rownames(RxC4) <- regions[[1]]
rownames(PxC1) <- proteins[[1]]
rownames(PxC2) <- proteins[[1]]
rownames(PxC3) <- proteins[[1]]
rownames(PxC4) <- proteins[[1]]
colnames(GxC1) <- rownames(meta_1)
colnames(GxC2) <- rownames(meta_2)
colnames(RxC3) <- rownames(meta_3)
colnames(RxC4) <- rownames(meta_4)
colnames(PxC1) <- rownames(meta_1)
colnames(PxC2) <- rownames(meta_2)
colnames(PxC3) <- rownames(meta_3)
colnames(PxC4) <- rownames(meta_4)

# create experiments
assay_list <- list("rna_b1" = GxC1, "rna_b2" = GxC2, "atac_b3" = RxC3, "atac_b4" = RxC4, "protein_b1" = PxC1, "protein_b2" = PxC2, "protein_b3" = PxC3, "protein_b4" = PxC4)
# one to many match, cell meta
meta <- rbind(meta_1, meta_2, meta_3, meta_4)
pbmc <- MultiAssayExperiment(experiments = assay_list, colData = meta)
# check relationship (validatioin)
upsetSamples(pbmc)
# check experiments (count matrices)
experiments(pbmc)


# running stabmap

# preprocessing of scRNA-seq
GxC <- SummarizedExperiment(list("counts" = cbind(GxC1, GxC2)), colData = rbind(meta_1, meta_2))
GxC <- logNormCounts(GxC)
# feature selection
decomp <- modelGeneVar(GxC)
hvgs <- rownames(decomp)[decomp$mean>0.01 & decomp$p.value <= 0.1]
length(hvgs)
GxC <- GxC[hvgs,]
# preprocessing of protein
PxC <- SummarizedExperiment(list("counts" = cbind(PxC1, PxC2, PxC3, PxC4)), colData = rbind(meta_1, meta_2, meta_3, meta_4))
PxC <- logNormCounts(PxC)
# preprocessing of atac
RxC <- SummarizedExperiment(list("counts" = cbind(RxC3, RxC4), "logcounts" = cbind(RxC3, RxC4)), colData = rbind(meta_3, meta_4))
# feature selection
decomp <- modelGeneVar(RxC)
hvgs <- rownames(decomp)[decomp$mean>0.01 & decomp$p.value <= 0.1]
length(hvgs)
RxC <- RxC[hvgs,]


GxC1 <- assays(GxC[, 1:dim(GxC1)[2]])[["logcounts"]]
GxC2 <- assays(GxC[, (dim(GxC1)[2] + 1):(dim(GxC1)[2] + dim(GxC2)[2])])[["logcounts"]]
PxC1 <- assays(PxC[, 1:dim(PxC1)[2]])[["logcounts"]]
PxC2 <- assays(PxC[, (dim(PxC1)[2] + 1):(dim(PxC1)[2] + dim(PxC2)[2])])[["logcounts"]]
PxC3 <- assays(PxC[, (dim(PxC1)[2] + dim(PxC2)[2] + 1):(dim(PxC1)[2] + dim(PxC2)[2] + dim(PxC3)[2])])[["logcounts"]]
PxC4 <- assays(PxC[, (dim(GxC1)[2] + dim(PxC2)[2] + dim(PxC3)[2] + 1):(dim(GxC1)[2] + dim(GxC2)[2] + dim(PxC3)[2] + dim(PxC4)[2])])[["logcounts"]]
RxC3 <- assays(RxC[, 1:dim(RxC3)[2]])[["logcounts"]]
RxC4 <- assays(RxC[, (dim(RxC3)[2] + 1):(dim(RxC3)[2] + dim(RxC4)[2])])[["logcounts"]]

# assays
assay_list = list("batch1" = rbind(GxC1, PxC1), "batch2" = rbind(GxC2, PxC2), "batch3" = rbind(RxC3, PxC3), "batch4" = rbind(RxC4, PxC4))
lapply(assay_list, dim)
# check relationship
mosaicDataUpSet(assay_list, plot = FALSE)
mdt = mosaicDataTopology(assay_list)
plot(mdt)

# running stabmap
stab = stabMap(assay_list, reference_list = c("batch1"), plot = FALSE)

# calculate UMAP
stab_umap = calculateUMAP(t(stab))
dim(stab_umap)
# PLOT RESULT
plot(stab_umap, pch = 16, cex = 0.3, col = factor(meta[["coarse_cluster"]]))

stab_b1 <- stab[1:dim(PxC1)[2], ]
stab_b2 <- stab[(dim(PxC1)[2] + 1):(dim(PxC1)[2] + dim(PxC2)[2]), ]
stab_b3 <- stab[(dim(PxC1)[2] + dim(PxC2)[2] + 1):(dim(PxC1)[2] + dim(PxC2)[2] + dim(PxC3)[2]), ]
stab_b4 <- stab[(dim(PxC1)[2] + dim(PxC2)[2] + dim(PxC3)[2] + 1):(dim(PxC1)[2] + dim(PxC2)[2] + dim(PxC3)[2] + dim(PxC4)[2]), ]

write.csv(stab_b1, file = paste0(results_dir, "stab_b1.csv"), quote = FALSE)
write.csv(stab_b2, file = paste0(results_dir, "stab_b2.csv"), quote = FALSE)
write.csv(stab_b3, file = paste0(results_dir, "stab_b3.csv"), quote = FALSE)
write.csv(stab_b4, file = paste0(results_dir, "stab_b4.csv"), quote = FALSE)
