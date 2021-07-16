library(Matrix)
library(rliger)
setwd("~/Dropbox/Research/Projects/CFRM/CFRM")
# install.packages("matrixStats")
library(matrixStats)

# read npy file
library(reticulate)
np <- import("numpy")

# data reading
# path <- "./test/results_multi/diag/"
# C0 <- np$load(paste0(path, "2b3c_sigma0.1_b1_1_0_0_C0.npy"))
# row.names(C0) <- paste("Cell_", 1:dim(C0)[1], sep = "")
# colnames(C0) <- paste("Factor_", 1:dim(C0)[2], sep = "")
# C1 <- np$load(paste0(path, "2b3c_sigma0.1_b1_1_0_0_C1.npy"))
# row.names(C1) <- paste("Cell_", 1:dim(C1)[1], sep = "")
# colnames(C1) <- paste("Factor_", 1:dim(C1)[2], sep = "")
# 
# # softmax numerically stable version
# softmax <- function (x) {
#     x2 <- exp(x - matrixStats::logSumExp(x))
#     return(x2)
# }
# 
# 
# C0 <- t(apply(C0, 1, softmax))
# C1 <- t(apply(C1, 1, softmax))

# if((!is.matrix(C0))||(!is.matrix(C1))){
#     stop("C0, C1 is not matrix")
# }
# if(dim(C0)[1] > dim(C1)[1]){
#     reference <- "C0"
# }else{
#     reference <- "C1"
# }
# Cs <- list("C0" = C0, "C1" = C1)
# 
# Cs_clust <- rliger::quantile_norm(Cs, ref_dataset = reference, refine.knn = TRUE)
# cluster <- Cs_clust[[2]]
# write.csv(cluster, paste0("./test/results_multi/diag/", substr(data, 1, nchar(data)-1), "_clust_id.csv"))

use_softmax <- T

# For diagonal
print("diagonal")
path <- "./test/results_multi/diag/"
datasets <- c('2b3c_sigma0.1_b2_1/', '2b3c_sigma0.1_b2_2/', '2b3c_sigma0.2_b2_1/', '2b3c_sigma0.2_b2_2/', '2b3c_sigma0.3_b2_1/', '2b3c_sigma0.3_b2_2/',
           '2b4c_sigma0.1_b2_1/', '2b4c_sigma0.1_b2_2/', '2b4c_sigma0.2_b2_1/', '2b4c_sigma0.2_b2_2/', '2b4c_sigma0.3_b2_1/', '2b4c_sigma0.3_b2_2/',
           '2b5c_sigma0.1_b2_1/', '2b5c_sigma0.1_b2_2/', '2b5c_sigma0.2_b2_1/', '2b5c_sigma0.2_b2_2/', '2b5c_sigma0.3_b2_1/', '2b5c_sigma0.3_b2_2/',
           '2b3c_sigma0.1_b1_1/', '2b3c_sigma0.1_b1_2/', '2b3c_sigma0.2_b1_1/', '2b3c_sigma0.2_b1_2/', '2b3c_sigma0.3_b1_1/', '2b3c_sigma0.3_b1_2/',
           '2b4c_sigma0.1_b1_1/', '2b4c_sigma0.1_b1_2/', '2b4c_sigma0.2_b1_1/', '2b4c_sigma0.2_b1_2/', '2b4c_sigma0.3_b1_1/', '2b4c_sigma0.3_b1_2/',
           '2b5c_sigma0.1_b1_1/', '2b5c_sigma0.1_b1_2/', '2b5c_sigma0.2_b1_1/', '2b5c_sigma0.2_b1_2/', '2b5c_sigma0.3_b1_1/', '2b5c_sigma0.3_b1_2/'
)

for(data in datasets){
    for(i in seq(0,2)){
        for(run in seq(0,4)){
            C0 <- np$load(paste0(path, substr(data, 1, nchar(data)-1), "_", i, "_", run, "_C0.npy"))
            row.names(C0) <- paste("Cell_", 1:dim(C0)[1], sep = "")
            colnames(C0) <- paste("Factor_", 1:dim(C0)[2], sep = "")
            C1 <- np$load(paste0(path, substr(data, 1, nchar(data)-1), "_", i, "_", run, "_C1.npy"))
            row.names(C1) <- paste("Cell_", 1:dim(C1)[1], sep = "")
            colnames(C1) <- paste("Factor_", 1:dim(C1)[2], sep = "")
            
            # find reference
            if(dim(C0)[1] > dim(C1)[1]){
                reference <- "C0"
            }else{
                reference <- "C1"
            }
            if(use_softmax){
                # softmax
                C0 <- t(apply(C0, 1, softmax))
                C1 <- t(apply(C1, 1, softmax))                
            }
            
            Cs <- list("C0" = C0, "C1" = C1)
            
            Cs_clust <- rliger::quantile_norm(Cs, min_cells = 1, max_sample = 2000, ref_dataset = reference, refine.knn = TRUE)
            cluster <- Cs_clust[[2]]
            write.csv(cluster, paste0("./test/results_multi/diag/", substr(data, 1, nchar(data)-1), "_", i, "_", run, "_clust_id.csv"))
            
        }        
    }
}


print("Horizontal")
path <- "./test/results_multi/hori/"
datasets <- c('2b3c_sigma0.1_b2_1/', '2b3c_sigma0.1_b2_2/', '2b3c_sigma0.2_b2_1/', '2b3c_sigma0.2_b2_2/', '2b3c_sigma0.3_b2_1/', '2b3c_sigma0.3_b2_2/',
              '2b4c_sigma0.1_b2_1/', '2b4c_sigma0.1_b2_2/', '2b4c_sigma0.2_b2_1/', '2b4c_sigma0.2_b2_2/', '2b4c_sigma0.3_b2_1/', '2b4c_sigma0.3_b2_2/',
              '2b5c_sigma0.1_b2_1/', '2b5c_sigma0.1_b2_2/', '2b5c_sigma0.2_b2_1/', '2b5c_sigma0.2_b2_2/', '2b5c_sigma0.3_b2_1/', '2b5c_sigma0.3_b2_2/',
              '2b3c_sigma0.1_b1_1/', '2b3c_sigma0.1_b1_2/', '2b3c_sigma0.2_b1_1/', '2b3c_sigma0.2_b1_2/', '2b3c_sigma0.3_b1_1/', '2b3c_sigma0.3_b1_2/',
              '2b4c_sigma0.1_b1_1/', '2b4c_sigma0.1_b1_2/', '2b4c_sigma0.2_b1_1/', '2b4c_sigma0.2_b1_2/', '2b4c_sigma0.3_b1_1/', '2b4c_sigma0.3_b1_2/',
              '2b5c_sigma0.1_b1_1/', '2b5c_sigma0.1_b1_2/', '2b5c_sigma0.2_b1_1/', '2b5c_sigma0.2_b1_2/', '2b5c_sigma0.3_b1_1/', '2b5c_sigma0.3_b1_2/'
)

for(data in datasets){
    for(i in seq(0,2)){
        for(run in seq(0,4)){
            C0 <- np$load(paste0(path, substr(data, 1, nchar(data)-1), "_", i, "_", run, "_C0.npy"))
            row.names(C0) <- paste("Cell_", 1:dim(C0)[1], sep = "")
            colnames(C0) <- paste("Factor_", 1:dim(C0)[2], sep = "")
            C1 <- np$load(paste0(path, substr(data, 1, nchar(data)-1), "_", i, "_", run, "_C1.npy"))
            row.names(C1) <- paste("Cell_", 1:dim(C1)[1], sep = "")
            colnames(C1) <- paste("Factor_", 1:dim(C1)[2], sep = "")
            
            # find reference
            if(dim(C0)[1] > dim(C1)[1]){
                reference <- "C0"
            }else{
                reference <- "C1"
            }
            
            if(use_softmax){
                # softmax
                C0 <- t(apply(C0, 1, softmax))
                C1 <- t(apply(C1, 1, softmax))                
            }
            
            Cs <- list("C0" = C0, "C1" = C1)
            
            Cs_clust <- rliger::quantile_norm(Cs, min_cells = 1, max_sample = 2000, ref_dataset = reference, refine.knn = TRUE)
            cluster <- Cs_clust[[2]]
            write.csv(cluster, paste0("./test/results_multi/hori/", substr(data, 1, nchar(data)-1), "_", i, "_", run, "_clust_id.csv"))
            
        }        
    }
}


print("Multi3")
path <- "./test/results_multi/multi3/"
datasets <- c('2b3c_sigma0.1_b2_1/', '2b3c_sigma0.1_b2_2/', '2b3c_sigma0.2_b2_1/', '2b3c_sigma0.2_b2_2/', '2b3c_sigma0.3_b2_1/', '2b3c_sigma0.3_b2_2/',
              '2b4c_sigma0.1_b2_1/', '2b4c_sigma0.1_b2_2/', '2b4c_sigma0.2_b2_1/', '2b4c_sigma0.2_b2_2/', '2b4c_sigma0.3_b2_1/', '2b4c_sigma0.3_b2_2/',
              '2b5c_sigma0.1_b2_1/', '2b5c_sigma0.1_b2_2/', '2b5c_sigma0.2_b2_1/', '2b5c_sigma0.2_b2_2/', '2b5c_sigma0.3_b2_1/', '2b5c_sigma0.3_b2_2/',
              '2b3c_sigma0.1_b1_1/', '2b3c_sigma0.1_b1_2/', '2b3c_sigma0.2_b1_1/', '2b3c_sigma0.2_b1_2/', '2b3c_sigma0.3_b1_1/', '2b3c_sigma0.3_b1_2/',
              '2b4c_sigma0.1_b1_1/', '2b4c_sigma0.1_b1_2/', '2b4c_sigma0.2_b1_1/', '2b4c_sigma0.2_b1_2/', '2b4c_sigma0.3_b1_1/', '2b4c_sigma0.3_b1_2/',
              '2b5c_sigma0.1_b1_1/', '2b5c_sigma0.1_b1_2/', '2b5c_sigma0.2_b1_1/', '2b5c_sigma0.2_b1_2/', '2b5c_sigma0.3_b1_1/', '2b5c_sigma0.3_b1_2/',
              '2b3c_sigma0.4_b1_1/', '2b3c_sigma0.4_b1_2/', '2b4c_sigma0.4_b1_1/', '2b4c_sigma0.4_b1_2/', '2b5c_sigma0.4_b1_1/', '2b5c_sigma0.4_b1_2/',
              '2b3c_sigma0.5_b1_1/', '2b3c_sigma0.5_b1_2/', '2b4c_sigma0.5_b1_1/', '2b4c_sigma0.5_b1_2/', '2b5c_sigma0.5_b1_1/', '2b5c_sigma0.5_b1_2/'
)

for(data in datasets){
    for(i in seq(0,2)){
        for(run in seq(0,4)){
            C0 <- np$load(paste0(path, substr(data, 1, nchar(data)-1), "_", i, "_", run, "_C0.npy"))
            row.names(C0) <- paste("Cell_", 1:dim(C0)[1], sep = "")
            colnames(C0) <- paste("Factor_", 1:dim(C0)[2], sep = "")
            C1 <- np$load(paste0(path, substr(data, 1, nchar(data)-1), "_", i, "_", run, "_C1.npy"))
            row.names(C1) <- paste("Cell_", 1:dim(C1)[1], sep = "")
            colnames(C1) <- paste("Factor_", 1:dim(C1)[2], sep = "")
            
            # find reference
            if(dim(C0)[1] > dim(C1)[1]){
                reference <- "C0"
            }else{
                reference <- "C1"
            }
            
            if(use_softmax){
                # softmax
                C0 <- t(apply(C0, 1, softmax))
                C1 <- t(apply(C1, 1, softmax))                
            }
            
            Cs <- list("C0" = C0, "C1" = C1)
            
            Cs_clust <- rliger::quantile_norm(Cs, min_cells = 1, max_sample = 2000, ref_dataset = reference, refine.knn = TRUE)
            cluster <- Cs_clust[[2]]
            write.csv(cluster, paste0("./test/results_multi/multi3/", substr(data, 1, nchar(data)-1), "_", i, "_", run, "_clust_id.csv"))
            
        }        
    }
}


print("Multi4")
path <- "./test/results_multi/multi4/"
datasets <- c('2b3c_sigma0.1_b2_1/', '2b3c_sigma0.1_b2_2/', '2b3c_sigma0.2_b2_1/', '2b3c_sigma0.2_b2_2/', '2b3c_sigma0.3_b2_1/', '2b3c_sigma0.3_b2_2/',
              '2b4c_sigma0.1_b2_1/', '2b4c_sigma0.1_b2_2/', '2b4c_sigma0.2_b2_1/', '2b4c_sigma0.2_b2_2/', '2b4c_sigma0.3_b2_1/', '2b4c_sigma0.3_b2_2/',
              '2b5c_sigma0.1_b2_1/', '2b5c_sigma0.1_b2_2/', '2b5c_sigma0.2_b2_1/', '2b5c_sigma0.2_b2_2/', '2b5c_sigma0.3_b2_1/', '2b5c_sigma0.3_b2_2/',
              '2b3c_sigma0.1_b1_1/', '2b3c_sigma0.1_b1_2/', '2b3c_sigma0.2_b1_1/', '2b3c_sigma0.2_b1_2/', '2b3c_sigma0.3_b1_1/', '2b3c_sigma0.3_b1_2/',
              '2b4c_sigma0.1_b1_1/', '2b4c_sigma0.1_b1_2/', '2b4c_sigma0.2_b1_1/', '2b4c_sigma0.2_b1_2/', '2b4c_sigma0.3_b1_1/', '2b4c_sigma0.3_b1_2/',
              '2b5c_sigma0.1_b1_1/', '2b5c_sigma0.1_b1_2/', '2b5c_sigma0.2_b1_1/', '2b5c_sigma0.2_b1_2/', '2b5c_sigma0.3_b1_1/', '2b5c_sigma0.3_b1_2/',
              '2b3c_sigma0.4_b1_1/', '2b3c_sigma0.4_b1_2/', '2b4c_sigma0.4_b1_1/', '2b4c_sigma0.4_b1_2/', '2b5c_sigma0.4_b1_1/', '2b5c_sigma0.4_b1_2/',
              '2b3c_sigma0.5_b1_1/', '2b3c_sigma0.5_b1_2/', '2b4c_sigma0.5_b1_1/', '2b4c_sigma0.5_b1_2/', '2b5c_sigma0.5_b1_1/', '2b5c_sigma0.5_b1_2/'
              
)

for(data in datasets){
    for(i in seq(0,2)){
        for(run in seq(0,4)){
            C0 <- np$load(paste0(path, substr(data, 1, nchar(data)-1), "_", i, "_", run, "_C0.npy"))
            row.names(C0) <- paste("Cell_", 1:dim(C0)[1], sep = "")
            colnames(C0) <- paste("Factor_", 1:dim(C0)[2], sep = "")
            C1 <- np$load(paste0(path, substr(data, 1, nchar(data)-1), "_", i, "_", run, "_C1.npy"))
            row.names(C1) <- paste("Cell_", 1:dim(C1)[1], sep = "")
            colnames(C1) <- paste("Factor_", 1:dim(C1)[2], sep = "")
            
            # find reference
            if(dim(C0)[1] > dim(C1)[1]){
                reference <- "C0"
            }else{
                reference <- "C1"
            }
            
            if(use_softmax){
                # softmax
                C0 <- t(apply(C0, 1, softmax))
                C1 <- t(apply(C1, 1, softmax))                
            }
            
            Cs <- list("C0" = C0, "C1" = C1)
            
            Cs_clust <- rliger::quantile_norm(Cs, min_cells = 1, max_sample = 2000, ref_dataset = reference, refine.knn = TRUE)
            cluster <- Cs_clust[[2]]
            write.csv(cluster, paste0("./test/results_multi/multi4/", substr(data, 1, nchar(data)-1), "_", i, "_", run, "_clust_id.csv"))
            
        }        
    }
}
