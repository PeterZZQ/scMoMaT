rm(list = ls())
gc()

setwd("/localscratch/ziqi/Symsim2")

seed <- 0
set.seed(seed)
# read in the protein count matrix
PxC <- readMM("cite_seq/PxC1.mtx")
# PxC2 <- readMM("cite_seq/PxC2.mtx")
# PxC <- as.matrix(cbind(PxC1, PxC2))
# GxC1 <- readMM("cite_seq/GxC1.mtx")
# GxC2 <- readMM("cite_seq/GxC2.mtx")
# GxC <- cbind(GxC1, GxC2)

protein.count <- as.vector(PxC)
# fit and sample from density function:
# 1. https://stackoverflow.com/questions/17087312/generate-a-random-number-from-a-density-object-or-more-broadly-from-a-set-of-nu
# 2. https://stat.ethz.ch/R-manual/R-devel/library/stats/html/density.html
fit <- density(protein.count, kernel = "gaussian")

sample_protein <- function(GxC, reference_counts, fit, protein.idx){
  ngenes <- dim(GxC)[1]
  ncells <- dim(GxC)[2]
  
  # generate samples
  PxC <- as.matrix(GxC[protein.idx,])
  N <- dim(PxC)[1] * dim(PxC)[2]
  protein.samples <- rnorm(N, sample(reference_counts, size = N, replace = T), fit$bw)
  protein.samples[protein.samples < 0] <- 0
  protein.samples <- sort(protein.samples, decreasing = F)
  # add permutation
  permutation.idx <- sample(seq(N), floor(0.1*N), replace = F)
  protein.samples[sort(permutation.idx, decreasing = F)] <- protein.samples[permutation.idx]

  # replace the matrix with the rank value
  rank.PxC <- PxC
  rank.PxC[] <- rank(PxC)
  new.PxC <- apply(rank.PxC, 2, function(x) protein.samples[x])
  new.PxC
}

# read in the gene expression data
data_path <- "../scMoMaT/data/simulated/6b16c_test_10/unequal/"
GxC1 <- as.matrix(read.table(paste0(data_path, 'GxC1.txt'), sep = "\t"))
GxC2 <- as.matrix(read.table(paste0(data_path, 'GxC2.txt'), sep = "\t"))
GxC3 <- as.matrix(read.table(paste0(data_path, 'GxC3.txt'), sep = "\t"))
GxC4 <- as.matrix(read.table(paste0(data_path, 'GxC4.txt'), sep = "\t"))
GxC5 <- as.matrix(read.table(paste0(data_path, 'GxC5.txt'), sep = "\t"))
GxC6 <- as.matrix(read.table(paste0(data_path, 'GxC6.txt'), sep = "\t"))


# sample the genes that associate to proteins (200 proteins <-> 200 genes)
nproteins <- 200
# select the top variable genes
GxC.var <- GxC1
ngenes <- dim(GxC.var)[1]
libsize <- as.vector(colSums(GxC.var, dim = 1))
GxC.norm <- t(t(GxC.var)/libsize)
GxC.norm <- log1p(GxC.norm)
protein.idx <- rank(-rowVars(GxC.norm))[1:nproteins] 
# or sample randomly
# protein.idx <- sample(ngenes, nproteins, replace = FALSE)

# obtain protein and gene relationship
PxG <- matrix(0, nproteins, ngenes)
for(protein in 1:dim(PxG)[1]){
  PxG[protein,protein.idx[protein]] <- 1 
}

# sample protein count
PxC1 <- sample_protein(GxC1, reference_counts = protein.count, fit = fit, protein.idx = protein.idx)
PxC2 <- sample_protein(GxC2, reference_counts = protein.count, fit = fit, protein.idx = protein.idx)
PxC3 <- sample_protein(GxC3, reference_counts = protein.count, fit = fit, protein.idx = protein.idx)
PxC4 <- sample_protein(GxC4, reference_counts = protein.count, fit = fit, protein.idx = protein.idx)
PxC5 <- sample_protein(GxC5, reference_counts = protein.count, fit = fit, protein.idx = protein.idx)
PxC6 <- sample_protein(GxC6, reference_counts = protein.count, fit = fit, protein.idx = protein.idx)


write.table(PxC1, sprintf("%s/PxC1.txt", data_path), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(PxC2, sprintf("%s/PxC2.txt", data_path), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(PxC3, sprintf("%s/PxC3.txt", data_path), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(PxC4, sprintf("%s/PxC4.txt", data_path), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(PxC5, sprintf("%s/PxC5.txt", data_path), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(PxC6, sprintf("%s/PxC6.txt", data_path), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(PxG, sprintf("%s/protein2gene.txt", data_path), quote=F, row.names = F, col.names = F, sep = "\t")



# sample cell-specific protein counts
# sample_protein2 <- function(GxC, PxC.ref, nproteins = 300){
#   # sample cells
#   ngenes <- dim(GxC)[1]
#   ncells <- dim(GxC)[2]
#   cell.idx <- sample(dim(PxC.ref)[2], ncells, replace = TRUE)
#   PxC.ref <- PxC.ref[,cell.idx]
# 
#   # sample proteins
#   # select the top variable genes
#   GxC.var <- GxC
#   libsize <- as.vector(colSums(GxC.var, dim = 1))
#   GxC.norm <- t(t(GxC.var)/libsize)
#   GxC.norm <- log1p(GxC.norm)
#   protein.idx <- rank(-rowVars(GxC.norm))[1:nproteins] 
#   # or sample randomly
#   # protein.idx <- sample(ngenes, nproteins, replace = FALSE)
#   
#   # generate samples
#   PxC <- as.matrix(GxC[protein.idx,])
#   rank.PxC <- apply(PxC, 2, function(x) rank(x))
#   for(cell.i in 1:dim(PxC)[2]){
#     # sample from reference protein counts
#     fit <- density(PxC.ref[, cell.i], kernel = "gaussian")
#     protein.samples <- rnorm(nproteins, sample(PxC.ref[, cell.i], size = nproteins, replace = T), fit$bw)
#     protein.samples <- sort(protein.samples, decreasing = F)
#     PxC[, cell.i]
#   }
#   rank.PxC
# }