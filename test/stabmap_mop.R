rm(list = ls())
gc()

library(StabMap)
library(SingleCellMultiModal)
library(scran)
library(MultiAssayExperiment)
library(scater)
setwd("/localscratch/ziqi/scMoMaT/test/")

dir <- "../data/real/MOp_5batches/"
results_dir <- "MOp_5batches/stabmap/"

counts.rna.1 <- readMM(paste0(dir, 'GxC1.mtx'))
counts.rna.2 <- readMM(paste0(dir, 'GxC2.mtx'))
counts.atac.1 <- readMM(paste0(dir, 'RxC1.mtx'))
counts.atac.3 <- readMM(paste0(dir, 'RxC3.mtx'))
counts.rna.4 <- readMM(paste0(dir, "GxC4.mtx"))
counts.atac.5 <- readMM(paste0(dir, "RxC5.mtx"))

genes <- read.csv(paste0(dir, "genes.txt"), header = F)[[1]]
regions <- read.csv(paste0(dir, "regions.txt"), header = F)[[1]]
meta.cell.1 <- read.csv(paste0(dir, "meta_c1.csv"), header = T, sep = ",", row.names = 1)
meta.cell.2 <- read.csv(paste0(dir, "meta_c2.csv"), header = T, sep = ",", row.names = 1)
meta.cell.3 <- read.csv(paste0(dir, "meta_c3.csv"), header = T, sep = ",", row.names = 1)
meta.cell.4 <- read.csv(paste0(dir, "meta_c4.csv"), header = T, sep = ",", row.names = 1)
meta.cell.5 <- read.csv(paste0(dir, "meta_c5.csv"), header = T, sep = ",", row.names = 1)
rownames(meta.cell.1) <- paste("batch_1:", rownames(meta.cell.1), sep = "")
rownames(meta.cell.2) <- paste("batch_1:", rownames(meta.cell.2), sep = "")
rownames(meta.cell.3) <- paste("batch_1:", rownames(meta.cell.3), sep = "")
rownames(meta.cell.4) <- paste("batch_1:", rownames(meta.cell.4), sep = "")
rownames(meta.cell.5) <- paste("batch_1:", rownames(meta.cell.5), sep = "")


rownames(counts.rna.1) <- genes
rownames(counts.rna.2) <- genes
rownames(counts.rna.4) <- genes
rownames(counts.atac.1) <- regions
rownames(counts.atac.3) <- regions
rownames(counts.atac.5) <- regions

colnames(counts.rna.1) <- rownames(meta.cell.1)
colnames(counts.rna.2) <- rownames(meta.cell.2)
colnames(counts.rna.4) <- rownames(meta.cell.4)
colnames(counts.atac.1) <- rownames(meta.cell.1)
colnames(counts.atac.3) <- rownames(meta.cell.3)
colnames(counts.atac.5) <- rownames(meta.cell.5)


# running stabmap
# preprocessing of scRNA-seq
GxC <- SummarizedExperiment(list("counts" = cbind(counts.rna.1, counts.rna.2, counts.rna.4)))
GxC <- logNormCounts(GxC)
# feature selection
decomp <- modelGeneVar(GxC)
hvgs <- rownames(decomp)[decomp$mean>0.01 & decomp$p.value <= 0.1]
length(hvgs)
GxC <- GxC[hvgs,]
# preprocessing of atac
RxC <- SummarizedExperiment(list("counts" = cbind(counts.atac.1, counts.atac.3, counts.atac.5), "logcounts" = cbind(counts.atac.1, counts.atac.3, counts.atac.5)))
# feature selection
decomp <- modelGeneVar(RxC)
hvgs <- rownames(decomp)[decomp$mean>0.01 & decomp$p.value <= 0.1]
length(hvgs)
RxC <- RxC[hvgs,]


GxC1 <- assays(GxC[, 1:dim(counts.rna.1)[2]])[["logcounts"]]
GxC2 <- assays(GxC[, (dim(counts.rna.1)[2] + 1):(dim(counts.rna.1)[2] + dim(counts.rna.2)[2])])[["logcounts"]]
GxC4 <- assays(GxC[, (dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + 1):(dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.4)[2])])[["logcounts"]]
RxC1 <- assays(RxC[, 1:dim(counts.atac.1)[2]])[["logcounts"]]
RxC3 <- assays(RxC[, (dim(counts.atac.1)[2] + 1):(dim(counts.atac.1)[2] + dim(counts.atac.3)[2])])[["logcounts"]]
RxC5 <- assays(RxC[, (dim(counts.atac.1)[2] + dim(counts.atac.3)[2] + 1):(dim(counts.atac.1)[2] + dim(counts.atac.3)[2] + dim(counts.atac.5)[2])])[["logcounts"]]

# assays
assay_list = list("batch1" = rbind(GxC1, RxC1), "batch2" = GxC2, "batch3" = RxC3, "batch4" = GxC4, "batch5" = RxC5)
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
# plot(stab_umap, pch = 16, cex = 0.3, col = factor(meta[["coarse_cluster"]]))

stab_b1 <- stab[1:dim(GxC1)[2], ]
stab_b2 <- stab[(dim(GxC1)[2] + 1):(dim(GxC1)[2] + dim(GxC2)[2]), ]
stab_b3 <- stab[(dim(GxC1)[2] + dim(GxC2)[2] + 1):(dim(GxC1)[2] + dim(GxC2)[2] + dim(RxC3)[2]), ]
stab_b4 <- stab[(dim(GxC1)[2] + dim(GxC2)[2] + dim(RxC3)[2] + 1):(dim(GxC1)[2] + dim(GxC2)[2] + dim(RxC3)[2] + dim(GxC4)[2]), ]
stab_b5 <- stab[(dim(GxC1)[2] + dim(GxC2)[2] + dim(RxC3)[2] + dim(GxC4)[2]):(dim(GxC1)[2] + dim(GxC2)[2] + dim(RxC3)[2] + dim(GxC4)[2] + dim(RxC5)[2]), ]

write.csv(stab_b1, file = paste0(results_dir, "stab_b1.csv"), quote = FALSE)
write.csv(stab_b2, file = paste0(results_dir, "stab_b2.csv"), quote = FALSE)
write.csv(stab_b3, file = paste0(results_dir, "stab_b3.csv"), quote = FALSE)
write.csv(stab_b4, file = paste0(results_dir, "stab_b4.csv"), quote = FALSE)
write.csv(stab_b5, file = paste0(results_dir, "stab_b5.csv"), quote = FALSE)
