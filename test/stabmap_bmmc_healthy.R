rm(list = ls())
gc()

library(StabMap)
library(SingleCellMultiModal)
library(scran)
library(MultiAssayExperiment)
library(scater)
setwd("/localscratch/ziqi/scMoMaT/test/")

path <- "../data/real/diag/healthy_hema/topgenes_1000/BMMC/"
result_path <- "bmmc_healthyhema_1000/stabmap/"

RxC1 <- readMM(paste0(path, "RxC1.mtx"))
rownames(RxC1) <- read.table(paste0(path, "regions.txt"), header = F, sep = ",")[[1]]
colnames(RxC1) <- rownames(read.table(paste0(path, "meta_c1.csv"), header = T, row.names = 1, sep = ","))

GxC2 <- readMM(paste0(path, "GxC2.mtx"))
rownames(GxC2) <- read.table(paste0(path, "genes.txt"), header = F, sep = ",")[[1]]
colnames(GxC2) <- rownames(read.table(paste0(path, "meta_c2.csv"), header = T, row.names = 1, sep = ","))

gene2region <- readMM(paste0(path, "GxR.mtx"))
rownames(gene2region) <- rownames(GxC2)
colnames(gene2region) <- rownames(RxC1)  

GxC1 <- gene2region %*% RxC1

# running stabmap

# preprocessing of scRNA-seq
GxC <- SummarizedExperiment(list("counts" = cbind(GxC1, GxC2)))
GxC <- logNormCounts(GxC)
# feature selection
decomp <- modelGeneVar(GxC)
hvgs <- rownames(decomp)[decomp$mean>0.01 & decomp$p.value <= 0.1]
length(hvgs)
GxC <- GxC[hvgs,]
# preprocessing of atac
RxC <- SummarizedExperiment(list("counts" = RxC1, "logcounts" = RxC1))
# feature selection
decomp <- modelGeneVar(RxC)
hvgs <- rownames(decomp)[decomp$mean>0.01 & decomp$p.value <= 0.1]
length(hvgs)
RxC <- RxC[hvgs,]


GxC1 <- assays(GxC[, 1:dim(GxC1)[2]])[["logcounts"]]
GxC2 <- assays(GxC[, (dim(GxC1)[2] + 1):(dim(GxC1)[2] + dim(GxC2)[2])])[["logcounts"]]
RxC1 <- assays(RxC[, 1:dim(RxC1)[2]])[["logcounts"]]

# assays
assay_list = list("batch1" = rbind(GxC1, RxC1), "batch2" = GxC2)
lapply(assay_list, dim)
# check relationship
mosaicDataUpSet(assay_list, plot = FALSE)
mdt = mosaicDataTopology(assay_list)
plot(mdt)

# running stabmap
stab = stabMap(assay_list, reference_list = c("batch1"), plot = FALSE)

# calculate UMAP
stab_umap = calculateUMAP(t(stab))

stab_b1 <- stab[1:dim(GxC1)[2], ]
stab_b2 <- stab[(dim(GxC1)[2] + 1):(dim(GxC1)[2] + dim(GxC2)[2]), ]

write.csv(stab_b1, file = paste0(result_path, "stab_b1.csv"), quote = FALSE)
write.csv(stab_b2, file = paste0(result_path, "stab_b2.csv"), quote = FALSE)
