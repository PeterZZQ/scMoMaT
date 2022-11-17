rm(list = ls())
gc()

library(StabMap)
library(SingleCellMultiModal)
library(scran)
library(MultiAssayExperiment)
library(scater)
setwd("/localscratch/ziqi/scMoMaT/test/")

# Read in the count matrix
path <- "../data/simulated/6b16c_test_10/imbalanced/"
result_path <- "simulated/6b16c_test_10/imbalanced/stabmap/"
dir.create(result_path, showWarnings = FALSE)

counts.rna.1 <- as.matrix(read.table(paste0(path, 'GxC1.txt'), sep = "\t"))
counts.rna.2 <- as.matrix(read.table(paste0(path, 'GxC2.txt'), sep = "\t"))
counts.rna.3 <- as.matrix(read.table(paste0(path, 'GxC3.txt'), sep = "\t"))
counts.rna.4 <- as.matrix(read.table(paste0(path, 'GxC4.txt'), sep = "\t"))
counts.rna.5 <- as.matrix(read.table(paste0(path, 'GxC5.txt'), sep = "\t"))
counts.rna.6 <- as.matrix(read.table(paste0(path, 'GxC6.txt'), sep = "\t"))

counts.atac.1 <- as.matrix(read.table(paste0(path, 'RxC1.txt'), sep = "\t"))
counts.atac.2 <- as.matrix(read.table(paste0(path, 'RxC2.txt'), sep = "\t"))
counts.atac.3 <- as.matrix(read.table(paste0(path, 'RxC3.txt'), sep = "\t"))
counts.atac.4 <- as.matrix(read.table(paste0(path, 'RxC4.txt'), sep = "\t"))
counts.atac.5 <- as.matrix(read.table(paste0(path, 'RxC5.txt'), sep = "\t"))
counts.atac.6 <- as.matrix(read.table(paste0(path, 'RxC6.txt'), sep = "\t"))

genes <- paste("gene_", seq(1, 1000), sep = "")
regions <- paste("region_", seq(1, 3000), sep = "")
meta.cell.1 <- read.csv(paste0(path, "cell_label1.txt"), header = T, sep = "\t", row.names = 1)
meta.cell.2 <- read.csv(paste0(path, "cell_label2.txt"), header = T, sep = "\t", row.names = 1)
meta.cell.3 <- read.csv(paste0(path, "cell_label3.txt"), header = T, sep = "\t", row.names = 1)
meta.cell.4 <- read.csv(paste0(path, "cell_label4.txt"), header = T, sep = "\t", row.names = 1)
meta.cell.5 <- read.csv(paste0(path, "cell_label5.txt"), header = T, sep = "\t", row.names = 1)
meta.cell.6 <- read.csv(paste0(path, "cell_label6.txt"), header = T, sep = "\t", row.names = 1)

rownames(counts.rna.1) <- genes
rownames(counts.rna.2) <- genes
rownames(counts.rna.3) <- genes
rownames(counts.rna.4) <- genes
rownames(counts.rna.5) <- genes
rownames(counts.rna.6) <- genes

rownames(counts.atac.1) <- regions
rownames(counts.atac.2) <- regions
rownames(counts.atac.3) <- regions
rownames(counts.atac.4) <- regions
rownames(counts.atac.5) <- regions
rownames(counts.atac.6) <- regions

colnames(counts.rna.1) <- rownames(meta.cell.1)
colnames(counts.rna.2) <- rownames(meta.cell.2)
colnames(counts.rna.3) <- rownames(meta.cell.3)
colnames(counts.rna.4) <- rownames(meta.cell.4)
colnames(counts.rna.5) <- rownames(meta.cell.5)
colnames(counts.rna.6) <- rownames(meta.cell.6)

colnames(counts.atac.1) <- rownames(meta.cell.1)
colnames(counts.atac.2) <- rownames(meta.cell.2)
colnames(counts.atac.3) <- rownames(meta.cell.3)
colnames(counts.atac.4) <- rownames(meta.cell.4)
colnames(counts.atac.5) <- rownames(meta.cell.5)
colnames(counts.atac.6) <- rownames(meta.cell.6)


gene2region <- t(as.matrix(read.table(paste0(path, "region2gene.txt"), sep = "\t")))
rownames(gene2region) <- genes
colnames(gene2region) <- regions


activity.matrix.1 <- gene2region %*% counts.atac.1
activity.matrix.2 <- gene2region %*% counts.atac.2
activity.matrix.3 <- gene2region %*% counts.atac.3
activity.matrix.4 <- gene2region %*% counts.atac.4
activity.matrix.5 <- gene2region %*% counts.atac.5
activity.matrix.6 <- gene2region %*% counts.atac.6


# preprocessing of scRNA-seq
GxC <- SummarizedExperiment(list("counts" = cbind(counts.rna.1, counts.rna.2, counts.rna.3, counts.rna.4, counts.rna.5, counts.rna.6)))
GxC <- logNormCounts(GxC)
# feature selection
decomp <- modelGeneVar(GxC)
hvgs <- rownames(decomp)[decomp$mean>0.01 & decomp$p.value <= 0.1]
length(hvgs)
GxC <- GxC[hvgs,]

# preprocessing of activity matrices
activity <- SummarizedExperiment(list("counts" = cbind(activity.matrix.1, activity.matrix.2, activity.matrix.3, activity.matrix.4, activity.matrix.5, activity.matrix.6)))
activity <- logNormCounts(activity)
activity <- activity[hvgs,]

# preprocessing of atac
RxC <- SummarizedExperiment(list("counts" = cbind(counts.atac.1, counts.atac.2, counts.atac.3, counts.atac.4, counts.atac.5, counts.atac.6), 
                                 "logcounts" = cbind(counts.atac.1, counts.atac.2, counts.atac.3, counts.atac.4, counts.atac.5, counts.atac.6)))
# feature selection
decomp <- modelGeneVar(RxC)
hvgs <- rownames(decomp)[decomp$mean>0.01 & decomp$p.value <= 0.1]
length(hvgs)
RxC <- RxC[hvgs,]


GxC1 <- assays(GxC[, 1:dim(counts.rna.1)[2]])[["logcounts"]]
GxC2 <- assays(GxC[, (dim(counts.rna.1)[2] + 1):(dim(counts.rna.1)[2] + dim(counts.rna.2)[2])])[["logcounts"]]
GxC3 <- assays(GxC[, (dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + 1):(dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2])])[["logcounts"]]
GxC4 <- assays(GxC[, (dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + 1):(dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2])])[["logcounts"]]
GxC5 <- assays(GxC[, (dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + 1):(dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + dim(counts.rna.5)[2])])[["logcounts"]]
GxC6 <- assays(GxC[, (dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + dim(counts.rna.5)[2] + 1):(dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + dim(counts.rna.5)[2] + dim(counts.rna.6)[2])])[["logcounts"]]

RxC1 <- assays(RxC[, 1:dim(counts.atac.1)[2]])[["logcounts"]]
RxC2 <- assays(RxC[, (dim(counts.atac.1)[2] + 1):(dim(counts.atac.1)[2] + dim(counts.atac.2)[2])])[["logcounts"]]
RxC3 <- assays(RxC[, (dim(counts.atac.1)[2] + dim(counts.atac.2)[2] + 1):(dim(counts.atac.1)[2] + dim(counts.atac.2)[2] + dim(counts.atac.3)[2])])[["logcounts"]]
RxC4 <- assays(RxC[, (dim(counts.atac.1)[2] + dim(counts.atac.2)[2] + dim(counts.atac.3)[2] + 1):(dim(counts.atac.1)[2] + dim(counts.atac.2)[2] + dim(counts.atac.3)[2] + dim(counts.atac.4)[2])])[["logcounts"]]
RxC5 <- assays(RxC[, (dim(counts.atac.1)[2] + dim(counts.atac.2)[2] + dim(counts.atac.3)[2] + dim(counts.atac.4)[2] + 1):(dim(counts.atac.1)[2] + dim(counts.atac.2)[2] + dim(counts.atac.3)[2] + dim(counts.atac.4)[2] + dim(counts.atac.5)[2])])[["logcounts"]]
RxC6 <- assays(RxC[, (dim(counts.atac.1)[2] + dim(counts.atac.2)[2] + dim(counts.atac.3)[2] + dim(counts.atac.4)[2] + dim(counts.atac.5)[2] + 1):(dim(counts.atac.1)[2] + dim(counts.atac.2)[2] + dim(counts.atac.3)[2] + dim(counts.atac.4)[2] + dim(counts.atac.5)[2] + dim(counts.atac.6)[2])])[["logcounts"]]

GxC1.activity <- assays(activity[, 1:dim(counts.rna.1)[2]])[["logcounts"]]
GxC2.activity <- assays(activity[, (dim(counts.rna.1)[2] + 1):(dim(counts.rna.1)[2] + dim(counts.rna.2)[2])])[["logcounts"]]
GxC3.activity <- assays(activity[, (dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + 1):(dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2])])[["logcounts"]]
GxC4.activity <- assays(activity[, (dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + 1):(dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2])])[["logcounts"]]
GxC5.activity <- assays(activity[, (dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + 1):(dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + dim(counts.rna.5)[2])])[["logcounts"]]
GxC6.activity <- assays(activity[, (dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + dim(counts.rna.5)[2] + 1):(dim(counts.rna.1)[2] + dim(counts.rna.2)[2] + dim(counts.rna.3)[2] + dim(counts.rna.4)[2] + dim(counts.rna.5)[2] + dim(counts.rna.6)[2])])[["logcounts"]]

# assays
# scenario 1
# assay_list = list("batch1" = rbind(RxC1, GxC1.activity), "batch2" = rbind(RxC2, GxC2.activity), "batch3" = rbind(RxC3, GxC3.activity), "batch4" = GxC4, "batch5" = GxC5, "batch6" = GxC6)
# scenario 2
assay_list = list("batch1" = rbind(RxC1, GxC1.activity), "batch2" = rbind(RxC2, GxC2.activity), "batch3" = rbind(RxC3, GxC3.activity), "batch4" = rbind(RxC4, GxC4), "batch5" = GxC5, "batch6" = GxC6)

lapply(assay_list, dim)
# check relationship
mosaicDataUpSet(assay_list, plot = FALSE)
mdt = mosaicDataTopology(assay_list)
plot(mdt)

# running stabmap, use all sample batches as reference performs bad
# normal
# stab = stabMap(assay_list, reference_list = c("batch1"), plot = FALSE)
# imbalanced, batch 1 is small
stab = stabMap(assay_list, reference_list = c("batch2"), plot = FALSE)

# calculate UMAP
# stab_umap = calculateUMAP(t(stab))

stab_b1 <- stab[1:dim(GxC1)[2], ]
stab_b2 <- stab[(dim(GxC1)[2] + 1):(dim(GxC1)[2] + dim(GxC2)[2]), ]
stab_b3 <- stab[(dim(GxC1)[2] + dim(GxC2)[2] + 1):(dim(GxC1)[2] + dim(GxC2)[2] + dim(GxC3)[2]), ]
stab_b4 <- stab[(dim(GxC1)[2] + dim(GxC2)[2] + dim(GxC3)[2] + 1):(dim(GxC1)[2] + dim(GxC2)[2] + dim(GxC3)[2] + dim(GxC4)[2]), ]
stab_b5 <- stab[(dim(GxC1)[2] + dim(GxC2)[2] + dim(GxC3)[2] + dim(GxC4)[2] + 1):(dim(GxC1)[2] + dim(GxC2)[2] + dim(GxC3)[2] + dim(GxC4)[2] + dim(GxC5)[2]), ]
stab_b6 <- stab[(dim(GxC1)[2] + dim(GxC2)[2] + dim(GxC3)[2] + dim(GxC4)[2] + dim(GxC5)[2] + 1):(dim(GxC1)[2] + dim(GxC2)[2] + dim(GxC3)[2] + dim(GxC4)[2] + dim(GxC5)[2] + dim(GxC6)[2]), ]

write.csv(stab_b1, file = paste0(result_path, "stab_b1.csv"), quote = FALSE)
write.csv(stab_b2, file = paste0(result_path, "stab_b2.csv"), quote = FALSE)
write.csv(stab_b3, file = paste0(result_path, "stab_b3.csv"), quote = FALSE)
write.csv(stab_b4, file = paste0(result_path, "stab_b4.csv"), quote = FALSE)
write.csv(stab_b5, file = paste0(result_path, "stab_b5.csv"), quote = FALSE)
write.csv(stab_b6, file = paste0(result_path, "stab_b6.csv"), quote = FALSE)
