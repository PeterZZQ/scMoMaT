library(Matrix)

num_clusters <- 9 #number true clusters in the data
number_of_matrices <- 2 #number of matrices you're loading
min_cells <- 20 #hyperparater for quantile alignment
max_sample <-  1000#hyperparater for quantile alignment
quantiles = 50#hyperparater for quantile alignment
path = "./"

C1 = read.table(file = paste0(path, "C1.txt"), sep = ",", header = None)
C2 = read.table(file = paste0(path, "C2.txt"), sep = ",", header = None)

cluster <- apply(C1,1,which.max), apply(C2,1,which.max)


#CHANGE CHANGE IF NUMBER OF MATRICES CHANGE, file U matrices
f1 <- dataset[[1]]
f2 <- dataset[[2]]
Hs <- list(f1, f2)


max_count = 0
ref_dataset = 1
for (k in 1:length(Hs)) {
    if(length(clusters[[k]]) > max_count){
        # reference dataset with maximum number of cells?
        ref_dataset = k
        max_count = length(clusters[[k]])
    }
}
dims = NCOL(f1)
if (number_of_matrices != length(clusters)){
    print("Incorrect matrix count")
}

#from quantile_norm function in https://rdrr.io/github/MacoskoLab/liger/src/R/liger.R

# loop through all matrices
for (k in 1:length(Hs)) {
    # number of clusters
    for (j in 1:num_clusters) {
        # the jth cluster
        index_for_comparision <- j
        # if not reference dataset
        if(use_association & k != ref_dataset){
            # get the cluster in current dataset that match reference cluster
            index_for_comparision <- closest_cluster_for_matrix2_in_1[j]
        }

        # find the cell correspond to the cluster in corrent dataset
        cells2 <- which(clusters[[k]] == index_for_comparision)
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

            q2 <- quantile(sample(Hs[[k]][cells2, i], min(num_cells2, max_sample)), seq(0, 1, by = 1 / quantiles))
            q1 <- quantile(sample(Hs[[ref_dataset]][cells1, i], min(num_cells1, max_sample)), seq(0, 1, by = 1 / quantiles))
            if (sum(q1) == 0 | sum(q2) == 0 | length(unique(q1)) <
                2 | length(unique(q2)) < 2) {
                new_vals <- rep(0, num_cells2)
            }
            else {
                warp_func <- stats::approxfun(q2, q1, rule = 2)
                new_vals <- warp_func(Hs[[k]][cells2, i])
            }
            Hs[[k]][cells2, i] <- new_vals
        }
    }
}
dump_result(outname, Hs)
print(paste("Completed: ", outname, sep=" "))

