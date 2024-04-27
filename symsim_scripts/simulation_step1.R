rm(list = ls())
gc()
library("devtools")
library("ape")
setwd("/localscratch/ziqi/Symsim2/")
load_all("SymSim2")
DivideBatches2 <- function(observed_counts_res, atacseq_noisy=NA, nbatch, batch_effect_size=1){
  # add batch effects to observed scRNA-seq counts
  # use different mean and same sd to generate the multiplicative factor for different gene in different batch
  observed_counts <- observed_counts_res[["counts"]]
  meta_cell <- observed_counts_res[["cell_meta"]]
  ncells <- dim(observed_counts)[2]; ngenes <- dim(observed_counts)[1]
  # sample and assign the batch ID for each cell within the datasets
  batchIDs <- sample(1:nbatch, ncells, replace = TRUE)
  # include the batch ID information into the meta_cell
  meta_cell2 <- data.frame(batch=batchIDs, stringsAsFactors = F)
  meta_cell <- cbind(meta_cell, meta_cell2)
  
  # do we still need this line?
  mean_matrix <- matrix(0, ngenes, nbatch)
  # generate a random normal distribution vector with mean 0 and variance 1 (length ngenes)
  gene_mean <- rnorm(ngenes, 0, 1)
  # for each genes, generate a random uniform distributed vector of length nbatches. The min of the distribution is gene_mean - batch_effect_size,
  # and the max is gene_mean + batch_effect_size. As a result, the batch_effect_size is controlling the variance of the mean_matrix. 
  temp <- lapply(1:ngenes, function(igene) {
    return(runif(nbatch, min = gene_mean[igene]-batch_effect_size, max = gene_mean[igene]+batch_effect_size))
  })
  # the mean matrix is a gene by batch matrix, which stores the mean of the batch_factor (sampled from gaussian) such that the mean values of each batch is the same
  mean_matrix <- do.call(rbind, temp)
  
  batch_factor <- matrix(0, ngenes, ncells)
  for (igene in 1:ngenes){
    for (icell in 1:ncells){
      batch_factor[igene, icell] <- rnorm(n=1, mean=mean_matrix[igene, batchIDs[icell]], sd=0.01)
    }
  }
  # add the batch_factor (batch effect) onto the count matrix
  observed_counts <- round(2^(log2(observed_counts)+batch_factor))
  
  # add batch effect to observed scATAC-seq counts, the mechanism is the same as scRNA-seq
  if (!all(is.na(atacseq_noisy))){
    nregions <- dim(atacseq_noisy)[1]
    n_batch_regions <- 100
    mean_matrix <- matrix(0, nregions, nbatch)
    region_mean <- rnorm(nregions, 0, 1)
    temp <- lapply(1:nregions, function(iregion) {
      if (iregion > n_batch_regions * nbatch){
        batch_effect_iregion <- runif(nbatch, min = region_mean[iregion]-batch_effect_size, max = region_mean[iregion]+batch_effect_size)
        # print("non batch specific region")
        # print(length(batch_effect_iregion))
        return(batch_effect_iregion)
      } else {
        # add batch specific regions
        ibatch <- (iregion - 1) %/%n_batch_regions + 1
        batch_effect_iregion <- rep(-batch_effect_size * 10, nbatch)
        batch_effect_iregion[ibatch] <- batch_effect_size * 3
        # print("batch specific regions")
        # print(ibatch)
        # print(length(batch_effect_iregion))
        return(batch_effect_iregion)
      }
        
    })
    mean_matrix <- do.call(rbind, temp)
    
    batch_factor <- matrix(0, nregions, ncells)
    for (iregion in 1:nregions){
      for (icell in 1:ncells){
        batch_factor[iregion, icell] <- rnorm(n=1, mean=mean_matrix[iregion, batchIDs[icell]], sd=0.01)
      }
    }
    atacseq_noisy <- round(2^(log2(atacseq_noisy)+batch_factor))
    
    return(list(scRNAseq=observed_counts, scATACseq=atacseq_noisy, cell_meta=meta_cell, mean_matrix=mean_matrix, batch_factor=batch_factor))
  } else {
    return(list(scRNAseq=observed_counts, cell_meta=meta_cell))
  }
}

getDEgenes2 <- function(true_counts_res, popA){
  meta_cell <- true_counts_res$cell_meta
  meta_gene <- true_counts_res$gene_effects
  popA_idx <- which(meta_cell$pop==popA)
  popB_idx <- which(meta_cell$pop!=popA)
  
  DEstr <- sapply(strsplit(colnames(meta_cell)[which(grepl("evf",colnames(meta_cell)))], "_"), "[[", 2)
  param_str <- sapply(strsplit(colnames(meta_cell)[which(grepl("evf",colnames(meta_cell)))], "_"), "[[", 1)
  n_useDEevf <- sapply(1:ngenes, function(igene) {
    return(sum(abs(meta_gene[[1]][igene, DEstr[which(param_str=="kon")]=="DE"])-0.001 > 0)+
             sum(abs(meta_gene[[2]][igene, DEstr[which(param_str=="koff")]=="DE"])-0.001 > 0)+
             sum(abs(meta_gene[[3]][igene, DEstr[which(param_str=="s")]=="DE"])-0.001 > 0))
  })
  
  kon_mat <- true_counts_res$kinetic_params[[1]]
  koff_mat <- true_counts_res$kinetic_params[[2]]
  s_mat <- true_counts_res$kinetic_params[[3]]
  
  logFC_theoretical <- sapply(1:ngenes, function(igene)
    return( log2(mean(s_mat[igene, popA_idx]*kon_mat[igene, popA_idx]/(kon_mat[igene, popA_idx]+koff_mat[igene, popA_idx]))/
                   mean(s_mat[igene, popB_idx]*kon_mat[igene, popB_idx]/(kon_mat[igene, popB_idx]+koff_mat[igene, popB_idx])) ) ))
  
  wil.p_theoretical <- sapply(1:ngenes, function(igene) 
    return(wilcox.test(s_mat[igene, popA_idx]*kon_mat[igene, popA_idx]/(kon_mat[igene, popA_idx]+koff_mat[igene, popA_idx]), 
                       s_mat[igene, popB_idx]*kon_mat[igene, popB_idx]/(kon_mat[igene, popB_idx]+koff_mat[igene, popB_idx]))$p.value))
  
  true_counts <- true_counts_res$counts
  true_counts_norm <- t(t(true_counts)/colSums(true_counts))*10^6
  
  wil.p_true_counts <- sapply(1:ngenes, function(igene) 
    return(wilcox.test(true_counts_norm[igene, popA_idx], true_counts_norm[igene, popB_idx])$p.value))
  
  wil.adjp_true_counts <- p.adjust(wil.p_true_counts, method = 'fdr')
  
  return(list(nDiffEVF=n_useDEevf, logFC_theoretical=logFC_theoretical, wil.p_true_counts=wil.p_true_counts, wil.p_theoretical=wil.p_theoretical))
}


PlotTsne <- function(meta, data, evf_type, pca = T, n_pc, perplexity=30, label, saving=F, plotname,system.color=T){
  uniqcols<-c(1:length(data[1,]))[!duplicated(t(data))]
  data <- data[,uniqcols];meta <- meta[uniqcols,,drop=FALSE] 
  uniqrows<-c(1:length(data[,1]))[!duplicated(data)]
  data <- data[uniqrows,]
  if (pca){
    data_pc <- prcomp(t(data))
    data <- t(data_pc$x[,c(1:n_pc)])
  }
  data_tsne=Rtsne(t(data),perplexity=perplexity)
  plot_tsne <- cbind(meta, data.frame(label=factor(meta[,label]),x=data_tsne$Y[,1],y=data_tsne$Y[,2]))
  p <- ggplot(plot_tsne, aes(x, y))
  p <- p + geom_point(aes(colour = .data[["label"]]),shape=20) + labs(color=label)
  if (system.color==F){
    if(evf_type=="discrete" | evf_type=="one.population"){
      color_5pop <- c("#F67670", "#0101FF", "#005826", "#A646F5", "#980000")
      names(color_5pop) <- 1:5
    }else{
      color_5pop <- c("#CC9521", "#1EBDC8", "#0101FF", "#005826", "#7CCC1F", "#A646F5", "#980000", "#F67670")
      names(color_5pop) <- c("6_7","7_8","8_2","8_3","7_9","9_4","9_5","6_1")
    }
    # p <- p + scale_color_manual(values=color_5pop[levels(plot_tsne[['label']])])
  }
  if(saving==T){ggsave(p,filename=plotname,device='pdf',width=5,height=4)}
  if(saving==F){p <- p + ggtitle(plotname)}
  return(list(plot_tsne,p))
}

########################################################################
#
# generate region to gene matrix
#
########################################################################

# A gene is regulated by k consecutive regions
# regions 1 to nregion are considered sequentially located on the genome
ngenes <- 1000
nregions <- 3000 
seed <- 7
print(seed)
# the probability that a gene is regulated by respectively 0, 1, 2 regions
p0 <- 0.01
# prob_REperGene <- c(p0, (1-p0)*(1/10), (1-p0)*(1/10),(1-p0)*(1/5), (1-p0)*(1/5),(1-p0)*(1/5),(1-p0)*(1/10),(1-p0)*(1/10))
prob_REperGene <- c(p0, (1-p0)*(1/5), (1-p0)*(4/5))

cumsum_prob <- cumsum(prob_REperGene)

region2gene <- matrix(0, nregions, ngenes)
set.seed(seed)
rand_vec <- runif(ngenes)

for (igene in 1:ngenes){
  if (rand_vec[igene] >= cumsum_prob[1] & rand_vec[igene] < cumsum_prob[2]) {
    region2gene[round(runif(1,min = 1, max = nregions)),igene] <- 1 
  } else if (rand_vec[igene] >= cumsum_prob[2]){
    startpos <- round(runif(1,min = 1, max = nregions-1))
    region2gene[startpos: (startpos+1),igene] <- c(1,1)
  }
}


ncells_total <- 10000
min_popsize <- 100
# ncells_total <- 10000
# min_popsize <- 2000

########################################################################
#
# Define trajectory structure
#
########################################################################
# 1 cluster
# phyla <- read.tree(text="(t1:1);")
# 2 clusters
# phyla <- read.tree(text="(t1:1,t2:0.5);")
# 3 clusters
# phyla <- read.tree(text="(t1:1,t2:1.5,t3:0.5);")
# 5 clusters
# phyla <- read.tree(text="(((A:1,B:1):1,(C:0.5,D:0.5):1.5):1,E:3);")
# 10 clusters
# phyla <- read.tree(text="(((A:1,B:1,F:1.5):1,(C:0.5,D:0.5,G:1.5):1.5,(H:0.5,I:0.5,J:1.5):2.0):1,E:3);")
# 16 clusters
phyla <- read.tree(text="(((A:1,B:1,F:1.5):1,(C:0.5,D:0.5,G:1.5):1.5,(H:0.5,I:0.5,J:1.5):2.0):1,((K:1,L:1,M:1.5):2.5,(N:0.5,O:0.5,P:1.5):3.0):2,E:3);")
# 20 clusters
# phyla <- read.tree(text="(((A:1,B:1,F:1.5):1,(C:0.5,D:0.5,G:1.5):1.5,(H:0.5,I:0.5,J:1.5):2.0):1,((K:1,L:1,M:1.5):2.5,(N:0.5,O:0.5,P:1.5):3.0):2,(R:0.5,S:1,T:0.75,U:2):1.5,E:3);")

plot(phyla)

########################################################################
#
# Simulate true scATAC-Seq and scRNA-Seq
#
########################################################################
# simulate the true count, the parameter setting the same as symsim
# update atac_effect, stronger the effect is, stronger the connection between atac and rna can be
true_counts_res <- SimulateTrueCounts(ncells_total=ncells_total,min_popsize=min_popsize,i_minpop=2,ngenes=dim(region2gene)[2], 
                                      nregions=dim(region2gene)[1],region2gene=region2gene,atac_effect=0.9,
                                      evf_center=1,evf_type="discrete",nevf=12,
                                      n_de_evf=8,n_de_evf_atac = 3, impulse=F,vary='s',Sigma=0.4,
                                      phyla=phyla,geffect_mean=0,gene_effects_sd=1,gene_effect_prob=0.3,
                                      bimod=0,param_realdata="zeisel.imputed",scale_s=0.8,
                                      prop_hge=0.015, mean_hge=5, randseed=seed, gene_module_prop=0)
atacseq_data <- true_counts_res[[2]]
rnaseq_data <- true_counts_res[[1]]

# plot the tsne of true scRNA-Seq count 
tsne_rnaseq_true <- PlotTsne(meta=true_counts_res[[4]], data=log2(rnaseq_data+1),
                             evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="true rna-seq")

tsne_atacseq_true <- PlotTsne(meta=true_counts_res[[4]], data=log2(atacseq_data+1),
                              evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="true atac-seq")

########################################################################
#
# Simulate technical noise
#
########################################################################
# generate observed scRNA-Seq count
data(gene_len_pool)
gene_len <- sample(gene_len_pool, ngenes, replace = FALSE)
observed_rnaseq <- True2ObservedCounts(true_counts=rnaseq_data, meta_cell=true_counts_res[[4]], 
                                       protocol="UMI", alpha_mean=0.2, alpha_sd=0.05, 
                                       gene_len=gene_len, depth_mean=5e5, depth_sd=3e3)

# cell_meta includes batch column
# observed_counts_loBE <- DivideBatches(observed_counts_res = observed_rnaseq, atacseq_noisy = atacseq_data, nbatch = 6, batch_effect_size = 1)
# observed_rnaseq_loBE <- observed_counts_loBE$scRNAseq
# atacseq_data <- observed_counts_loBE$scATACseq
# meta_cell <- observed_counts_loBE$cell_meta

# generated observed scATAC-Seq count
atacseq_data <- round(atacseq_data)
atacseq_noisy <- atacseq_data

# # add noise
# for (icell in 1:ncells_total){
#   for (iregion in 1:nregions){
#     if (atacseq_data[iregion, icell] > 0){
#       # downsample the count (atacseq_data[iregion, icell]) with probability 0.3
#       atacseq_noisy[iregion, icell] <- rbinom(n=1, size = atacseq_data[iregion, icell], prob = 0.3)}
#     if (atacseq_noisy[iregion, icell] > 0){
#       atacseq_noisy[iregion, icell] <- atacseq_noisy[iregion, icell]+rnorm(1, mean = 0, sd=atacseq_noisy[iregion, icell]/10)
#     }
#   }
# }

# # introduce dropout
# atacseq_noisy[atacseq_noisy<0.1] <- 0
# # prop_1 calculate the proportion of non-zero elements in atacseq_noisy
# prop_1 <- sum(atacseq_noisy>0.1)/(dim(atacseq_noisy)[1]*dim(atacseq_noisy)[2])
# target_prop_1 <- 0.1
# # if the proportion of non-zero elements is larger than target_prop_1, then we 
# if (prop_1 > target_prop_1) { # need to set larger threshold to have more non-zero values become 0s
#   # calculate the number of count elements that need to be set 0
#   n2set0 <- ceiling((prop_1 - target_prop_1)*dim(atacseq_data)[1]*dim(atacseq_data)[2])
#   # calculate the threshold from the number of cells
#   threshold <- sort(atacseq_noisy[atacseq_noisy>0.1])[n2set0]
#   # set the values to be 0
#   atacseq_noisy[atacseq_noisy<threshold] <- 0
# } else {
#   print(sprintf("The proportion of 1s is %4.3f", prop_1))
# }

# observed_atacseq_loBE <- atacseq_noisy
# observed_counts_loBE <- DivideBatches(observed_counts_res = observed_rnaseq, atacseq_noisy = NA, nbatch = 6, batch_effect_size = 1)
# observed_rnaseq_loBE <- observed_counts_loBE$scRNAseq
# meta_cell <- observed_counts_loBE$cell_meta

observed_counts_loBE <- DivideBatches(observed_counts_res = observed_rnaseq, atacseq_noisy = atacseq_noisy, nbatch = 6, batch_effect_size = 1)
observed_rnaseq_loBE <- observed_counts_loBE$scRNAseq
observed_atacseq_loBE <- observed_counts_loBE$scATACseq
meta_cell <- observed_counts_loBE$cell_meta

# index of cells in each batch
cellset_batch1 <- which(meta_cell$batch==1)
cellset_batch2 <- which(meta_cell$batch==2)
cellset_batch3 <- which(meta_cell$batch==3)
cellset_batch4 <- which(meta_cell$batch==4)
cellset_batch5 <- which(meta_cell$batch==5)
cellset_batch6 <- which(meta_cell$batch==6)


# binarize the atacseq matrix
observed_atacseq_loBE[observed_atacseq_loBE > 0] <- 1
# Plot
tsne_rnaseq_noisy <- PlotTsne(meta=meta_cell, data=log2(observed_rnaseq_loBE+1), 
                              evf_type="discrete", n_pc=20, label='batch', saving = F, plotname="noisy rna-seq batches")
tsne_rnaseq_noisy2 <- PlotTsne(meta=meta_cell, data=log2(observed_rnaseq_loBE+1), 
                              evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="noisy rna-seq clusters")
tsne_atacseq_noisy <- PlotTsne(meta=meta_cell, data=observed_atacseq_loBE, 
                               evf_type="discrete", n_pc=20, label='batch', saving = F, plotname="noisy atac-seq batches")
tsne_atacseq_noisy2 <- PlotTsne(meta=meta_cell, data=observed_atacseq_loBE, 
                               evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="noisy atac-seq clusters")

########################################################################
#
# Save the result
#
########################################################################
for(popA in seq(1,16)){
  de_result <- getDEgenes2(true_counts_res = true_counts_res, popA = popA)
  de_result <- as.data.frame(de_result)
  de_result$pop <- popA
  if(popA == 1){
    de_results <- de_result
  }else{
    de_results <- rbind(de_results, de_result)
  }
}

datapath <- sprintf("./de_test_%d", seed)
# datapath <- "./de_test_5"
system(sprintf("mkdir %s", datapath))
write.table(region2gene, sprintf("%s/region2gene.txt", datapath),
            quote=F, row.names = F, col.names = F, sep = "\t")
write.table(observed_rnaseq_loBE[, cellset_batch1], sprintf("%s/GxC1.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(observed_rnaseq_loBE[, cellset_batch2], sprintf("%s/GxC2.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(observed_rnaseq_loBE[, cellset_batch3], sprintf("%s/GxC3.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(observed_rnaseq_loBE[, cellset_batch4], sprintf("%s/GxC4.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(observed_rnaseq_loBE[, cellset_batch5], sprintf("%s/GxC5.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(observed_rnaseq_loBE[, cellset_batch6], sprintf("%s/GxC6.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

write.table(observed_atacseq_loBE[, cellset_batch1], sprintf("%s/RxC1.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(observed_atacseq_loBE[, cellset_batch2], sprintf("%s/RxC2.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(observed_atacseq_loBE[, cellset_batch3], sprintf("%s/RxC3.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(observed_atacseq_loBE[, cellset_batch4], sprintf("%s/RxC4.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(observed_atacseq_loBE[, cellset_batch5], sprintf("%s/RxC5.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(observed_atacseq_loBE[, cellset_batch6], sprintf("%s/RxC6.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

write.table(rnaseq_data[, cellset_batch1], sprintf("%s/GxC1_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(rnaseq_data[, cellset_batch2], sprintf("%s/GxC2_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(rnaseq_data[, cellset_batch3], sprintf("%s/GxC3_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(rnaseq_data[, cellset_batch4], sprintf("%s/GxC4_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(rnaseq_data[, cellset_batch5], sprintf("%s/GxC5_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(rnaseq_data[, cellset_batch6], sprintf("%s/GxC6_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

write.table(atacseq_data[, cellset_batch1], sprintf("%s/RxC1_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(atacseq_data[, cellset_batch2], sprintf("%s/RxC2_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(atacseq_data[, cellset_batch3], sprintf("%s/RxC3_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(atacseq_data[, cellset_batch4], sprintf("%s/RxC4_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(atacseq_data[, cellset_batch5], sprintf("%s/RxC5_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
write.table(atacseq_data[, cellset_batch6], sprintf("%s/RxC6_true.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

write.table(true_counts_res[[4]][cellset_batch1,1:2], sprintf("%s/cell_label1.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")
write.table(true_counts_res[[4]][cellset_batch2,1:2], sprintf("%s/cell_label2.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")
write.table(true_counts_res[[4]][cellset_batch3,1:2], sprintf("%s/cell_label3.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")
write.table(true_counts_res[[4]][cellset_batch4,1:2], sprintf("%s/cell_label4.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")
write.table(true_counts_res[[4]][cellset_batch5,1:2], sprintf("%s/cell_label5.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")
write.table(true_counts_res[[4]][cellset_batch6,1:2], sprintf("%s/cell_label6.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")

write.table(de_results, sprintf("%s/de_genes.txt", datapath), quote=F, row.names = F, col.names = T, sep = "\t")

# write.table(mean_matrix, sprintf("%s/mean_matrix.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
# write.table(batch_factor, sprintf("%s/batch_factor.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

# write.table(true_counts_res[[4]][cellset_batch1,3], sprintf("%s/pseudotime1.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
# write.table(true_counts_res[[4]][cellset_batch2,3], sprintf("%s/pseudotime2.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
# write.table(true_counts_res[[4]][cellset_batch3,3], sprintf("%s/pseudotime3.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")
# write.table(true_counts_res[[4]][cellset_batch4,3], sprintf("%s/pseudotime4.txt", datapath), quote=F, row.names = F, col.names = F, sep = "\t")

# save the plots
pdf(file=sprintf("%s/tsne.pdf", datapath))
print(tsne_rnaseq_true[[2]])
print(tsne_atacseq_true[[2]])
print(tsne_rnaseq_noisy[[2]])
print(tsne_atacseq_noisy[[2]])
print(tsne_rnaseq_noisy2[[2]])
print(tsne_atacseq_noisy2[[2]])
dev.off()





