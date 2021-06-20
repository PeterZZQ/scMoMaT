library(Matrix)

num_clusters <- 9 #number true clusters in the data
number_of_matrices <- 2 #number of matrices you're loading
min_cells <- 20 #hyperparater for quantile alignment
max_sample <-  1000#hyperparater for quantile alignment
quantiles = 50#hyperparater for quantile alignment
path = "./src/"

C1 = read.table(file = paste0(path, "C1.txt"), sep = ",", header = FALSE, row.names = 1)
C2 = read.table(file = paste0(path, "C2.txt"), sep = ",", header = FALSE, row.names = 1)

cluster1 <- apply(C1, 1, which.max)
cluster2 <- apply(C2, 1, which.max)
clusters <- list(cluster1, cluster2)

Hs <- list(C1, C2)

max_count = 0
ref_dataset = 1
for (k in 1:length(Hs)) {
    if(length(clusters[[k]]) > max_count){
        # reference dataset with maximum number of cells?
        ref_dataset = k
        max_count = length(clusters[[k]])
    }
}
dims = NCOL(C1)
if (number_of_matrices != length(clusters)){
    print("Incorrect matrix count")
}

#from quantile_norm function in https://rdrr.io/github/MacoskoLab/liger/src/R/liger.R

# loop through all matrices
for (k in 1:length(Hs)) {
    # number of clusters
    for (j in 1:num_clusters) {

        # find the cell correspond to the jth cluster in current dataset
        cells2 <- which(clusters[[k]] == j)
        # find the cell correspond to the cluster in reference dataset
        cells1 <- which(clusters[[ref_dataset]] == j)

        # dims latent dimension
        for (i in 1:dims) {
            # number of cells
            num_cells2 <- length(cells2)
            num_cells1 <- length(cells1)

            # if number of cells less thant quantile min, skip
            if (num_cells1 < min_cells | num_cells2 < min_cells) {
                next
            }
            # if there is only one cells, calculate the mean as its assignment
            if (num_cells2 == 1) {
                Hs[[k]][cells2, i] <- mean(Hs[[ref_dataset]][cells1, i])
                next
            }

            # numpy.quantile
            q2 <- quantile(sample(Hs[[k]][cells2, i], min(num_cells2, max_sample)), seq(0, 1, by = 1 / quantiles))
            q1 <- quantile(sample(Hs[[ref_dataset]][cells1, i], min(num_cells1, max_sample)), seq(0, 1, by = 1 / quantiles))
            if (sum(q1) == 0 | sum(q2) == 0 | length(unique(q1)) <
                2 | length(unique(q2)) < 2) {
                new_vals <- rep(0, num_cells2)
            }
            else {
                # check scipy.interp1d
                # from scipy.interpolate import interp1d
                # f = interp1d(q2, q1)
                # new_vals = f(Hs[[k]][cells2, i])
                warp_func <- stats::approxfun(q2, q1, rule = 2)
                new_vals <- warp_func(Hs[[k]][cells2, i])
            }
            Hs[[k]][cells2, i] <- new_vals
        }
    }
}
dump_result(outname, Hs)
print(paste("Completed: ", outname, sep=" "))



# Original
# quantile_norm.list <- function(
#     object,
#     quantiles = 50,
#     ref_dataset = NULL,
#     min_cells = 20,
#     knn_k = 20,
#     dims.use = NULL,
#     do.center = FALSE,
#     max_sample = 1000,
#     eps = 0.9,
#     refine.knn = TRUE,
#     rand.seed = 1,
#     ...
# ) {
#   set.seed(rand.seed)
#   if (!all(sapply(X = object, FUN = is.matrix))) {
#     stop("All values in 'object' must be a matrix")
#   }
#   if (is.null(x = names(x = object))) {
#     stop("'object' must be a named list of matrices")
#   }
#   if (is.character(x = ref_dataset) && !ref_dataset %in% names(x = object)) {
#     stop("Cannot find reference dataset")
#   } else if (!inherits(x = ref_dataset, what = c('character', 'numeric'))) {
#     stop("'ref_dataset' must be a character or integer specifying which dataset is the reference")
#   }
#   labels <- list()
#   if (is.null(dims.use)) {
#     use_these_factors <- 1:ncol(object[[1]])
#   } else {
#     use_these_factors <- dims.use
#   }
#   # fast max factor assignment with Rcpp code
#   labels <- lapply(object, max_factor, dims_use = use_these_factors, center_cols = do.center)
#   clusters <- as.factor(unlist(lapply(labels, as.character)))
#   names(clusters) <- unlist(lapply(object, rownames))

#   # increase robustness of cluster assignments using knn graph
#   if (refine.knn) {
#     clusters <- refine_clusts_knn(object, clusters, k = knn_k, eps = eps)
#   }
#   cluster_assignments <- clusters
#   clusters <- lapply(object, function(x) {
#     clusters[rownames(x)]
#   })
#   names(clusters) <- names(object)
#   dims <- ncol(object[[ref_dataset]])

#   dataset <- unlist(lapply(1:length(object), function(i) {
#     rep(names(object)[i], nrow(object[[i]]))
#   }))
#   Hs <- object
#   num_clusters <- dims
#   for (k in 1:length(object)) {
#     for (j in 1:num_clusters) {
#       cells2 <- which(clusters[[k]] == j)
#       cells1 <- which(clusters[[ref_dataset]] == j)
#       for (i in 1:dims) {
#         num_cells2 <- length(cells2)
#         num_cells1 <- length(cells1)
#         if (num_cells1 < min_cells | num_cells2 < min_cells) {
#           next
#         }
#         if (num_cells2 == 1) {
#           Hs[[k]][cells2, i] <- mean(Hs[[ref_dataset]][cells1, i])
#           next
#         }
#         q2 <- quantile(sample(Hs[[k]][cells2, i], min(num_cells2, max_sample)), seq(0, 1, by = 1 / quantiles))
#         q1 <- quantile(sample(Hs[[ref_dataset]][cells1, i], min(num_cells1, max_sample)), seq(0, 1, by = 1 / quantiles))
#         if (sum(q1) == 0 | sum(q2) == 0 | length(unique(q1)) <
#           2 | length(unique(q2)) < 2) {
#           new_vals <- rep(0, num_cells2)
#         }
#         else {
#           warp_func <- withCallingHandlers(stats::approxfun(q2, q1, rule = 2), warning=function(w){invokeRestart("muffleWarning")})
#           new_vals <- warp_func(Hs[[k]][cells2, i])
#         }
#         Hs[[k]][cells2, i] <- new_vals
#       }
#     }
#   }
#   out <- list(
#     'H.norm' = Reduce(rbind, Hs),
#     'clusters' = cluster_assignments
#   )
#   return(out)
# }


