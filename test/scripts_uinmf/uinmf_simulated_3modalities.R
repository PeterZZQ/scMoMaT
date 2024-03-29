rm(list = ls())
gc()

library(rliger)
library(Seurat)
library(stringr)

# tutorial see: http://htmlpreview.github.io/?https://github.com/welch-lab/liger/blob/master/vignettes/SNAREseq_walkthrough.html
setwd("/localscratch/ziqi/scMoMaT/test/scripts_uinmf/")
# Read in the count matrix
data_dir <- "../../data/simulated/6b16c_test_10/unequal/"
results_dir <- "../simulated/6b16c_test_10/protein_scenario1/uinmf/"
dir.create(results_dir, showWarnings = FALSE)

counts.rna.1 <- as.matrix(read.table(paste0(data_dir, 'GxC1.txt'), sep = "\t"))
counts.rna.2 <- as.matrix(read.table(paste0(data_dir, 'GxC2.txt'), sep = "\t"))
counts.rna.3 <- as.matrix(read.table(paste0(data_dir, 'GxC3.txt'), sep = "\t"))
counts.rna.4 <- as.matrix(read.table(paste0(data_dir, 'GxC4.txt'), sep = "\t"))
counts.rna.5 <- as.matrix(read.table(paste0(data_dir, 'GxC5.txt'), sep = "\t"))
counts.rna.6 <- as.matrix(read.table(paste0(data_dir, 'GxC6.txt'), sep = "\t"))

counts.atac.1 <- as.matrix(read.table(paste0(data_dir, 'RxC1.txt'), sep = "\t"))
counts.atac.2 <- as.matrix(read.table(paste0(data_dir, 'RxC2.txt'), sep = "\t"))
counts.atac.3 <- as.matrix(read.table(paste0(data_dir, 'RxC3.txt'), sep = "\t"))
counts.atac.4 <- as.matrix(read.table(paste0(data_dir, 'RxC4.txt'), sep = "\t"))
counts.atac.5 <- as.matrix(read.table(paste0(data_dir, 'RxC5.txt'), sep = "\t"))
counts.atac.6 <- as.matrix(read.table(paste0(data_dir, 'RxC6.txt'), sep = "\t"))

counts.protein.1 <- as.matrix(read.table(paste0(data_dir, 'PxC1.txt'), sep = "\t"))
counts.protein.2 <- as.matrix(read.table(paste0(data_dir, 'PxC2.txt'), sep = "\t"))
counts.protein.3 <- as.matrix(read.table(paste0(data_dir, 'PxC3.txt'), sep = "\t"))
counts.protein.4 <- as.matrix(read.table(paste0(data_dir, 'PxC4.txt'), sep = "\t"))
counts.protein.5 <- as.matrix(read.table(paste0(data_dir, 'PxC5.txt'), sep = "\t"))
counts.protein.6 <- as.matrix(read.table(paste0(data_dir, 'PxC6.txt'), sep = "\t"))

genes <- paste("gene_", seq(1, 1000), sep = "")
regions <- paste("region_", seq(1, 3000), sep = "")
proteins <- paste("protein_", seq(1, 200), sep = "")

meta.cell.1 <- read.csv(paste0(data_dir, "cell_label1.txt"), header = T, sep = "\t", row.names = 1)
meta.cell.2 <- read.csv(paste0(data_dir, "cell_label2.txt"), header = T, sep = "\t", row.names = 1)
meta.cell.3 <- read.csv(paste0(data_dir, "cell_label3.txt"), header = T, sep = "\t", row.names = 1)
meta.cell.4 <- read.csv(paste0(data_dir, "cell_label4.txt"), header = T, sep = "\t", row.names = 1)
meta.cell.5 <- read.csv(paste0(data_dir, "cell_label5.txt"), header = T, sep = "\t", row.names = 1)
meta.cell.6 <- read.csv(paste0(data_dir, "cell_label6.txt"), header = T, sep = "\t", row.names = 1)

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

rownames(counts.protein.1) <- proteins
rownames(counts.protein.2) <- proteins
rownames(counts.protein.3) <- proteins
rownames(counts.protein.4) <- proteins
rownames(counts.protein.5) <- proteins
rownames(counts.protein.6) <- proteins

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

colnames(counts.protein.1) <- rownames(meta.cell.1)
colnames(counts.protein.2) <- rownames(meta.cell.2)
colnames(counts.protein.3) <- rownames(meta.cell.3)
colnames(counts.protein.4) <- rownames(meta.cell.4)
colnames(counts.protein.5) <- rownames(meta.cell.5)
colnames(counts.protein.6) <- rownames(meta.cell.6)


gene2region <- t(as.matrix(read.table(paste0(data_dir, "region2gene.txt"), sep = "\t")))
protein2gene <- as.matrix(read.table(paste0(data_dir, "protein2gene.txt"), sep = "\t"))
rownames(gene2region) <- genes
colnames(gene2region) <- regions
rownames(protein2gene) <- proteins
colnames(protein2gene) <- genes

#############################################################
#
# scenario 1
#
#############################################################

activity.matrix.1 <- protein2gene %*% gene2region %*% counts.atac.1
activity.matrix.2 <- protein2gene %*% gene2region %*% counts.atac.2
activity.matrix.5 <- protein2gene %*% counts.rna.5
activity.matrix.6 <- protein2gene %*% counts.rna.6

# # create the liger object for the unshared region data

# create the liger object for the unshared bin data
liger_bin <- createLiger(list(peak1 = counts.atac.1, peak2 = counts.atac.2), remove.missing = FALSE)
liger_bin <- rliger::normalize(liger_bin)

# use all regions as variable regions
top_regions <- unlist(regions)
liger_bin <- selectGenes(liger_bin)
liger_bin@var.genes <- top_regions
liger_bin <- scaleNotCenter(liger_bin)
unshared_atac1 <- liger_bin@scale.data$peak1
unshared_atac2 <- liger_bin@scale.data$peak2

liger_rna <- createLiger(list(rna5 = counts.rna.5, rna6 = counts.rna.6), remove.missing = FALSE)
liger_rna <- rliger::normalize(liger_rna)

# use all regions as variable regions
top_genes <- unlist(genes)
liger_rna <- selectGenes(liger_rna)
liger_rna@var.genes <- top_genes
liger_rna <- scaleNotCenter(liger_rna)
unshared_rna5 <- liger_rna@scale.data$rna5
unshared_rna6 <- liger_rna@scale.data$rna6

################################################
#
# Using UINMF
#
################################################

# create the liger object and normalize the shared data
ifnb_liger <- createLiger(list(protein1 = activity.matrix.1, protein2 = activity.matrix.2, 
                              protein3 = counts.protein.3, protein4 = counts.protein.4, 
                              protein5 =activity.matrix.5, protein6 = activity.matrix.6), remove.missing = FALSE)
ifnb_liger <- rliger::normalize(ifnb_liger)
# for protein we don't select has there are only 216 proteins
ifnb_liger <- selectGenes(ifnb_liger)
# scale the data
ifnb_liger <- scaleNotCenter(ifnb_liger)

# Add the unshared features that have been properly selected, such that they are added as a genes by cells matrix.
ifnb_liger@var.unshared.features[[1]] = colnames(unshared_atac1)
ifnb_liger@scale.unshared.data[[1]] = t(unshared_atac1)
ifnb_liger@var.unshared.features[[2]] = colnames(unshared_atac2)
ifnb_liger@scale.unshared.data[[2]] = t(unshared_atac2)
ifnb_liger@var.unshared.features[[5]] = colnames(unshared_rna5)
ifnb_liger@scale.unshared.data[[5]] = t(unshared_rna5)
ifnb_liger@var.unshared.features[[6]] = colnames(unshared_rna6)
ifnb_liger@scale.unshared.data[[6]] = t(unshared_rna6)
# Joint matrix factorization
ifnb_liger <- optimizeALS(ifnb_liger, k=12, use.unshared = TRUE, max_iters =30,thresh=1e-10)


ifnb_liger <- quantile_norm(ifnb_liger)
H1 <- ifnb_liger@H$protein1
H2 <- ifnb_liger@H$protein2
H3 <- ifnb_liger@H$protein3
H4 <- ifnb_liger@H$protein4
H5 <- ifnb_liger@H$protein5
H6 <- ifnb_liger@H$protein6


H.norm1 <- ifnb_liger@H.norm[1:dim(H1)[1],]
H.norm2 <- ifnb_liger@H.norm[(dim(H1)[1] + 1):(dim(H1)[1] + dim(H2)[1]),]
H.norm3 <- ifnb_liger@H.norm[(dim(H1)[1] + dim(H2)[1] + 1):(dim(H1)[1] + dim(H2)[1] + dim(H3)[1]),]
H.norm4 <- ifnb_liger@H.norm[(dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + 1):(dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1]),]
H.norm5 <- ifnb_liger@H.norm[(dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1] + 1):(dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1] + dim(H5)[1]),]
H.norm6 <- ifnb_liger@H.norm[(dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1] + dim(H5)[1] + 1):(dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1] + dim(H5)[1] + dim(H6)[1]),]

write.csv(H1, paste0(results_dir, "H1.csv"))
write.csv(H2, paste0(results_dir, "H2.csv"))
write.csv(H3, paste0(results_dir, "H3.csv"))
write.csv(H4, paste0(results_dir, "H4.csv"))
write.csv(H5, paste0(results_dir, "H5.csv"))
write.csv(H6, paste0(results_dir, "H6.csv"))

write.csv(H.norm1, paste0(results_dir, "H1_norm.csv"))
write.csv(H.norm2, paste0(results_dir, "H2_norm.csv"))
write.csv(H.norm3, paste0(results_dir, "H3_norm.csv"))
write.csv(H.norm4, paste0(results_dir, "H4_norm.csv"))
write.csv(H.norm5, paste0(results_dir, "H5_norm.csv"))
write.csv(H.norm6, paste0(results_dir, "H6_norm.csv"))


#############################################################
#
# scenario 2
#
#############################################################

results_dir <- "../simulated/6b16c_test_10/protein_scenario2/uinmf/"
dir.create(results_dir, showWarnings = FALSE)

activity.matrix.1 <- protein2gene %*% gene2region %*% counts.atac.1
activity.matrix.2 <- protein2gene %*% gene2region %*% counts.atac.2
activity.matrix.5 <- protein2gene %*% gene2region %*% counts.atac.5
activity.matrix.6 <- protein2gene %*% gene2region %*% counts.atac.6

# # create the liger object for the unshared region data

# create the liger object for the unshared bin data
liger_bin <- createLiger(list(peak1 = counts.atac.1, peak2 = counts.atac.2, peak3 = counts.atac.3), remove.missing = FALSE)
liger_bin <- rliger::normalize(liger_bin)

# use all regions as variable regions
top_regions <- unlist(regions)
liger_bin <- selectGenes(liger_bin)
liger_bin@var.genes <- top_regions
liger_bin <- scaleNotCenter(liger_bin)
unshared_atac1 <- liger_bin@scale.data$peak1
unshared_atac2 <- liger_bin@scale.data$peak2
unshared_atac3 <- liger_bin@scale.data$peak3

liger_rna <- createLiger(list(rna4 = counts.rna.4, rna5 = counts.rna.5, rna6 = counts.rna.6), remove.missing = FALSE)
liger_rna <- rliger::normalize(liger_rna)

# use all regions as variable regions
top_genes <- unlist(genes)
liger_rna <- selectGenes(liger_rna)
liger_rna@var.genes <- top_genes
liger_rna <- scaleNotCenter(liger_rna)
unshared_rna4 <- liger_rna@scale.data$rna4
unshared_rna5 <- liger_rna@scale.data$rna5
unshared_rna6 <- liger_rna@scale.data$rna6

################################################
#
# Using UINMF
#
################################################

# create the liger object and normalize the shared data
ifnb_liger <- createLiger(list(protein1 = activity.matrix.1, protein2 = activity.matrix.2, 
                               protein3 = counts.protein.3, protein4 = counts.protein.4, 
                               protein5 =activity.matrix.5, protein6 = activity.matrix.6), remove.missing = FALSE)
ifnb_liger <- rliger::normalize(ifnb_liger)
# for protein we don't select has there are only 216 proteins
ifnb_liger <- selectGenes(ifnb_liger)
# scale the data
ifnb_liger <- scaleNotCenter(ifnb_liger)

# Add the unshared features that have been properly selected, such that they are added as a genes by cells matrix.
ifnb_liger@var.unshared.features[[1]] = colnames(unshared_atac1)
ifnb_liger@scale.unshared.data[[1]] = t(unshared_atac1)
ifnb_liger@var.unshared.features[[2]] = colnames(unshared_atac2)
ifnb_liger@scale.unshared.data[[2]] = t(unshared_atac2)
ifnb_liger@var.unshared.features[[3]] = colnames(unshared_atac3)
ifnb_liger@scale.unshared.data[[3]] = t(unshared_atac3)
ifnb_liger@var.unshared.features[[4]] = colnames(unshared_rna4)
ifnb_liger@scale.unshared.data[[4]] = t(unshared_rna4)
ifnb_liger@var.unshared.features[[5]] = colnames(unshared_rna5)
ifnb_liger@scale.unshared.data[[5]] = t(unshared_rna5)
ifnb_liger@var.unshared.features[[6]] = colnames(unshared_rna6)
ifnb_liger@scale.unshared.data[[6]] = t(unshared_rna6)
# Joint matrix factorization
ifnb_liger <- optimizeALS(ifnb_liger, k=12, use.unshared = TRUE, max_iters =30,thresh=1e-10)


ifnb_liger <- quantile_norm(ifnb_liger)
H1 <- ifnb_liger@H$protein1
H2 <- ifnb_liger@H$protein2
H3 <- ifnb_liger@H$protein3
H4 <- ifnb_liger@H$protein4
H5 <- ifnb_liger@H$protein5
H6 <- ifnb_liger@H$protein6


H.norm1 <- ifnb_liger@H.norm[1:dim(H1)[1],]
H.norm2 <- ifnb_liger@H.norm[(dim(H1)[1] + 1):(dim(H1)[1] + dim(H2)[1]),]
H.norm3 <- ifnb_liger@H.norm[(dim(H1)[1] + dim(H2)[1] + 1):(dim(H1)[1] + dim(H2)[1] + dim(H3)[1]),]
H.norm4 <- ifnb_liger@H.norm[(dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + 1):(dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1]),]
H.norm5 <- ifnb_liger@H.norm[(dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1] + 1):(dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1] + dim(H5)[1]),]
H.norm6 <- ifnb_liger@H.norm[(dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1] + dim(H5)[1] + 1):(dim(H1)[1] + dim(H2)[1] + dim(H3)[1] + dim(H4)[1] + dim(H5)[1] + dim(H6)[1]),]

write.csv(H1, paste0(results_dir, "H1.csv"))
write.csv(H2, paste0(results_dir, "H2.csv"))
write.csv(H3, paste0(results_dir, "H3.csv"))
write.csv(H4, paste0(results_dir, "H4.csv"))
write.csv(H5, paste0(results_dir, "H5.csv"))
write.csv(H6, paste0(results_dir, "H6.csv"))

write.csv(H.norm1, paste0(results_dir, "H1_norm.csv"))
write.csv(H.norm2, paste0(results_dir, "H2_norm.csv"))
write.csv(H.norm3, paste0(results_dir, "H3_norm.csv"))
write.csv(H.norm4, paste0(results_dir, "H4_norm.csv"))
write.csv(H.norm5, paste0(results_dir, "H5_norm.csv"))
write.csv(H.norm6, paste0(results_dir, "H6_norm.csv"))