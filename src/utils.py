import numpy as np
import torch
# from torch_sparse import SparseTensor
import matplotlib.pyplot as plt


import numpy as np
from scipy import stats

import torch.nn.functional as F

# ----------------------------------------------------- # 

# Preprocessing

# ----------------------------------------------------- # 

def quantile_norm(X):
    """Normalize the columns of X to each have the same distribution.

    Given an expression matrix (microarray data, read counts, etc) of M genes
    by N samples, quantile normalization ensures all samples have the same
    spread of data (by construction).

    The data across each row are averaged to obtain an average column. Each
    column quantile is replaced with the corresponding quantile of the average
    column.

    Parameters
    ----------
    X : 2D array of float, shape (M, N)
        The input data, with M rows (genes/features) and N columns (samples).

    Returns
    -------
    Xn : 2D array of float, shape (M, N)
        The normalized data.
    """
    # compute the quantiles
    quantiles = np.mean(np.sort(X, axis=0), axis=1)

    # compute the column-wise ranks. Each observation is replaced with its
    # rank in that column: the smallest observation is replaced by 1, the
    # second-smallest by 2, ..., and the largest by M, the number of rows.
    ranks = np.apply_along_axis(stats.rankdata, 0, X)

    # convert ranks to integer indices from 0 to M-1
    rank_indices = ranks.astype(int) - 1

    # index the quantiles for each rank with the ranks matrix
    Xn = quantiles[rank_indices]

    return(Xn)

def quantile_norm_log(X):
    logX = np.log1p(X)
    logXn = quantile_norm(logX)
    return logXn

def preprocess_old(counts, mode = "standard", modality = "RNA"):
    """\
    Description:
    ------------
    Preprocess the dataset
    """
    if mode == "standard":
        if modality == "ATAC":
            counts = (counts > 0).astype(np.float)
            
        # normalize according to the library size
        libsize = np.median(np.sum(counts, axis = 1))
        counts = counts / np.sum(counts, axis = 1)[:,None] * libsize

        if modality == "RNA":
            counts = np.log1p(counts)

    elif mode == "quantile":
        if modality == "RNA":
            counts = quantile_norm_log(counts)
        else:
            # make binary
            counts = (counts > 0).astype(np.float) 
            # counts = counts / np.sum(counts, axis = 1)[:,None]
    
    elif mode == "gact":
        # gene by region matrix
        counts = counts/(np.sum(counts, axis = 1)[:,None] + 1e-6)


    return counts

def preprocess(counts, modality = "RNA"):
    """\
    Description:
    ------------
    Preprocess the dataset, for count, interaction matrices
    """
    if modality == "ATAC":
        # make binary
        counts = (counts > 0).astype(np.float) 

    elif modality == "interaction":
        # gene by region matrix
        counts = counts/(np.sum(counts, axis = 1)[:,None] + 1e-6)
    
    else:
        # other cases, e.g. Protein, RNA, etc
        counts = quantile_norm_log(counts)
        counts = counts/np.max(counts)

    return counts


# ----------------------------------------------------- # 

# Plot

# ----------------------------------------------------- # 


def plot_latent(z1, z2, anno1 = None, anno2 = None, mode = "joint", save = None, figsize = (20,10), axis_label = "Latent", **kwargs):
    """\
    Description
        Plot latent space
    Parameters
        z1
            the latent space of first data batch, of the shape (n_samples, n_dimensions)
        z2
            the latent space of the second data batch, of the shape (n_samples, n_dimensions)
        anno1
            the cluster annotation of the first data batch, of the  shape (n_samples,)
        anno2
            the cluster annotation of the second data batch, of the  shape (n_samples,)
        mode
            "joint": plot two latent spaces(from two batches) into one figure
            "separate" plot two latent spaces separately
        save
            file name for the figure
        figsize
            figure size
    """
    _kwargs = {
        "s": 10,
        "alpha": 0.9,
    }
    _kwargs.update(kwargs)

    fig = plt.figure(figsize = figsize)
    if mode == "modality":
        colormap = plt.cm.get_cmap("tab10")
        ax = fig.add_subplot()
        ax.scatter(z1[:,0], z1[:,1], color = colormap(1), label = "scRNA-Seq", **_kwargs)
        ax.scatter(z2[:,0], z2[:,1], color = colormap(2), label = "scATAC-Seq", **_kwargs)
        ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
        
        ax.tick_params(axis = "both", which = "major", labelsize = 15)

        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  

    elif mode == "joint":
        ax = fig.add_subplot()
        cluster_types = set([x for x in np.unique(anno1)]).union(set([x for x in np.unique(anno2)]))
        colormap = plt.cm.get_cmap("tab20", len(cluster_types))

        for i, cluster_type in enumerate(cluster_types):
            index = np.where(anno1 == cluster_type)[0]
            index2 = np.where(anno2 == cluster_type)[0]
            ax.scatter(np.concatenate((z1[index,0], z2[index2,0])), np.concatenate((z1[index,1],z2[index2,1])), color = colormap(i), label = cluster_type, **_kwargs)
        
        ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
        
        ax.tick_params(axis = "both", which = "major", labelsize = 15)

        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  


    elif mode == "separate":
        axs = fig.subplots(1,2)
        cluster_types = set([x for x in np.unique(anno1)]).union(set([x for x in np.unique(anno2)]))
        colormap = plt.cm.get_cmap("tab20", len(cluster_types))

        
        for i, cluster_type in enumerate(cluster_types):
            index = np.where(anno1 == cluster_type)[0]

            if index.shape[0] != 0:
                axs[0].scatter(z1[index,0], z1[index,1], color = colormap(i), label = cluster_type, **_kwargs)
        # axs[0].legend(fontsize = font_size)
        axs[0].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(0.94, 1), markerscale=4)
        axs[0].set_title("scRNA-Seq", fontsize = 25)

        axs[0].tick_params(axis = "both", which = "major", labelsize = 15)

        axs[0].set_xlabel(axis_label + " 1", fontsize = 19)
        axs[0].set_ylabel(axis_label + " 2", fontsize = 19)
        axs[0].set_xlim(np.min(np.concatenate((z1[:,0], z2[:,0]))), np.max(np.concatenate((z1[:,0], z2[:,0]))))
        axs[0].set_ylim(np.min(np.concatenate((z1[:,1], z2[:,1]))), np.max(np.concatenate((z1[:,1], z2[:,1]))))
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False)  


        for i, cluster_type in enumerate(cluster_types):
            index = np.where(anno2 == cluster_type)[0]

            if index.shape[0] != 0:
                axs[1].scatter(z2[index,0], z2[index,1], color = colormap(i), label = cluster_type, **_kwargs)
        # axs[1].axis("off")
        axs[1].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale=4)
        axs[1].set_title("scATAC-Seq", fontsize = 25)

        axs[1].tick_params(axis = "both", which = "major", labelsize = 15)

        axs[1].set_xlabel(axis_label + " 1", fontsize = 19)
        axs[1].set_ylabel(axis_label + " 2", fontsize = 19)
        axs[1].set_xlim(np.min(np.concatenate((z1[:,0], z2[:,0]))), np.max(np.concatenate((z1[:,0], z2[:,0]))))
        axs[1].set_ylim(np.min(np.concatenate((z1[:,1], z2[:,1]))), np.max(np.concatenate((z1[:,1], z2[:,1]))))
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        
    
    elif mode == "hybrid":
        axs = fig.subplots(1,2)
        cluster_types = set([x for x in np.unique(anno1)]).union(set([x for x in np.unique(anno2)]))
        colormap = plt.cm.get_cmap("tab20", len(cluster_types))

        for i, cluster_type in enumerate(cluster_types):
            index = np.where(anno1 == cluster_type)[0]
            index2 = np.where(anno2 == cluster_type)[0]
            axs[1].scatter(np.concatenate((z1[index,0], z2[index2,0])), np.concatenate((z1[index,1],z2[index2,1])), color = colormap(i), label = cluster_type, **_kwargs)
        
        axs[1].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale=4)
        
        axs[1].tick_params(axis = "both", which = "major", labelsize = 15)

        axs[1].set_xlabel(axis_label + " 1", fontsize = 19)
        axs[1].set_ylabel(axis_label + " 2", fontsize = 19)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)          

        colormap = plt.cm.get_cmap("tab10")
        axs[0].scatter(z1[:,0], z1[:,1], color = colormap(1), label = "scRNA-Seq", **_kwargs)
        axs[0].scatter(z2[:,0], z2[:,1], color = colormap(2), label = "scATAC-Seq", **_kwargs)
        axs[0].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(0.74, 1), markerscale=4)
        
        axs[0].tick_params(axis = "both", which = "major", labelsize = 15)

        axs[0].set_xlabel(axis_label + " 1", fontsize = 19)
        axs[0].set_ylabel(axis_label + " 2", fontsize = 19)
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False) 
        
        
    if save:
        fig.savefig(save, bbox_inches = "tight")

def plot_latent_ext(zs, annos = None, mode = "joint", save = None, figsize = (20,10), axis_label = "Latent", **kwargs):
    """\
    Description
        Plot latent space
    Parameters
        z1
            the latent space of first data batch, of the shape (n_samples, n_dimensions)
        z2
            the latent space of the second data batch, of the shape (n_samples, n_dimensions)
        anno1
            the cluster annotation of the first data batch, of the  shape (n_samples,)
        anno2
            the cluster annotation of the second data batch, of the  shape (n_samples,)
        mode
            "joint": plot two latent spaces(from two batches) into one figure
            "separate" plot two latent spaces separately
        save
            file name for the figure
        figsize
            figure size
    """
    _kwargs = {
        "s": 10,
        "alpha": 0.9,
    }
    _kwargs.update(kwargs)

    fig = plt.figure(figsize = figsize)
    if mode == "modality":
        colormap = plt.cm.get_cmap("tab20", len(zs))
        ax = fig.add_subplot()
        
        for batch in range(len(zs)):
            ax.scatter(zs[batch][:,0], zs[batch][:,1], color = colormap(batch), label = "batch " + str(batch), **_kwargs)
        ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
        ax.tick_params(axis = "both", which = "major", labelsize = 15)

        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  

    elif mode == "joint":
        ax = fig.add_subplot()
        cluster_types = set()
        for batch in range(len(zs)):
            cluster_types = cluster_types.union(set([x for x in np.unique(annos[batch])]))
        colormap = plt.cm.get_cmap("tab20", len(cluster_types))

        for i, cluster_type in enumerate(cluster_types):
            z_clust = []
            for batch in range(len(zs)):
                index = np.where(annos[batch] == cluster_type)[0]
                z_clust.append(zs[batch][index,:])
            ax.scatter(np.concatenate(z_clust, axis = 0)[:,0], np.concatenate(z_clust, axis = 0)[:,1], color = colormap(i), label = cluster_type, **_kwargs)
        
        ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
        
        ax.tick_params(axis = "both", which = "major", labelsize = 15)

        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  


    elif mode == "separate":
        axs = fig.subplots(len(zs),1)
        cluster_types = set()
        for batch in range(len(zs)):
            cluster_types = cluster_types.union(set([x for x in np.unique(annos[batch])]))
        colormap = plt.cm.get_cmap("tab20", len(cluster_types))


        for batch in range(len(zs)):
            z_clust = []
            for i, cluster_type in enumerate(cluster_types):
                index = np.where(annos[batch] == cluster_type)[0]
                axs[batch].scatter(zs[batch][index,0], zs[batch][index,1], color = colormap(i), label = cluster_type, **_kwargs)
            
            axs[batch].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(0.94, 1), markerscale=4)
            axs[batch].set_title("batch " + str(batch), fontsize = 25)

            axs[batch].tick_params(axis = "both", which = "major", labelsize = 15)

            axs[batch].set_xlabel(axis_label + " 1", fontsize = 19)
            axs[batch].set_ylabel(axis_label + " 2", fontsize = 19)
            # axs[batch].set_xlim(np.min(np.concatenate((z1[:,0], z2[:,0]))), np.max(np.concatenate((z1[:,0], z2[:,0]))))
            # axs[batch].set_ylim(np.min(np.concatenate((z1[:,1], z2[:,1]))), np.max(np.concatenate((z1[:,1], z2[:,1]))))
            axs[batch].spines['right'].set_visible(False)
            axs[batch].spines['top'].set_visible(False)  
        
        
    if save:
        fig.savefig(save, bbox_inches = "tight")


# ----------------------------------------------------- # 

# Post-processing steps

# ----------------------------------------------------- # 
def _pairwise_distances(x, y = None):
    x_norm = (x**2).sum(1).view(-1, 1)
    # calculate the pairwise distance between two datasets
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag)

    return torch.clamp(dist, 0.0, np.inf)

def match_embed(z_rna, z_atac, k = 10):
    # note that the distance is squared version
    dist = _pairwise_distances(z_atac, z_rna).numpy()
    knn_index = np.argpartition(dist, kth = k - 1, axis = 1)[:,(k-1)]
    kth_dist = np.take_along_axis(dist, knn_index[:,None], axis = 1)
    
    K = dist/(100 * kth_dist + 1e-6) 
    K = (dist <= kth_dist) * np.exp(-K) 
    K = K/np.sum(K, axis = 1)[:,None]

    z_atac = torch.FloatTensor(K).mm(z_rna)
    return z_rna, z_atac

def match_clust(z_rna, z_atac, k = 10, scale = 1):
    # note that the distance is squared version
    clust_rna = np.argmax(z_rna, axis = 1).squeeze()
    clust_atac = np.argmax(z_atac, axis = 1).squeeze()
    for clust in range(np.max((z_rna.shape[1], z_atac.shape[1]))):
        rna_idx = np.where(clust_rna == clust)[0]
        atac_idx = np.where(clust_atac == clust)[0]
        
        if (rna_idx.shape[0] != 0) and (atac_idx.shape[0] != 0):
            z_rna_clust = z_rna[rna_idx,:]
            z_atac_clust = z_atac[atac_idx,:]

            
            dist = _pairwise_distances(z_atac_clust, z_rna_clust).numpy()
            knn_index = np.argpartition(dist, kth = k - 1, axis = 1)[:,(k-1)]
            kth_dist = np.take_along_axis(dist, knn_index[:,None], axis = 1)

            K = dist/(kth_dist * scale)
            K = (dist <= kth_dist) * np.exp(-K) 
            K = K/np.sum(K, axis = 1)[:,None]

            z_atac[atac_idx,:] = torch.FloatTensor(K).mm(z_rna_clust)
        else:
            print("no cell in cluster {:d}".format(clust))
    return z_rna, z_atac    

def match_embed_seurat(z_rna, z_atac, k = 20):
    """\
        Seurat MNN style 
    """

    n_rna = z_rna.shape[0]
    n_atac = z_atac.shape[0]

    # mutual nearest neighbor
    dist1 = _pairwise_distances(z_atac, z_rna).numpy()
    knn_index = np.argpartition(dist1, kth = k - 1, axis = 1)[:,(k-1)]
    kth_dist = np.take_along_axis(dist1, knn_index[:,None], axis = 1)
    knn1 = (dist1 <= kth_dist)

    dist2 = _pairwise_distances(z_rna, z_atac).numpy()
    knn_index = np.argpartition(dist2, kth = k - 1, axis = 1)[:,(k-1)]
    kth_dist = np.take_along_axis(dist2, knn_index[:,None], axis = 1)
    knn2 = (dist2 <= kth_dist)
    
    assert knn1.shape[0] == knn2.shape[1]
    assert knn1.shape[1] == knn2.shape[0]
    knn = knn1 * knn2.T
    dist = knn * dist1

    # scoring, calculate shared nearest neighbor
    dist3 = _pairwise_distances(z_rna, z_rna).numpy()
    knn_index = np.argpartition(dist3, kth = k - 1, axis = 1)[:,(k-1)]
    kth_dist = np.take_along_axis(dist3, knn_index[:,None], axis = 1)
    knn3 = (dist3 <= kth_dist)
    
    dist4 = _pairwise_distances(z_atac, z_atac).numpy()
    knn_index = np.argpartition(dist4, kth = k - 1, axis = 1)[:,(k-1)]
    kth_dist = np.take_along_axis(dist4, knn_index[:,None], axis = 1)
    knn4 = (dist4 <= kth_dist)

    # shape (n_rna + n_atac, n_rna + n_atac)
    snn1 = np.concatenate((knn3, knn2), axis = 1)
    snn2 = np.concatenate((knn1, knn4), axis = 1)
    rows, cols = np.where(knn != 0)

    snn_counts = np.zeros_like(knn)
    for row, col in zip(rows, cols):
        # row correspond to atac, col correspond to rna
        n_snn = np.sum(snn1[col, :] * snn2[row,:])
        snn_counts[row, col] = n_snn
    
    snn_counts = snn_counts/np.max(snn_counts+ 1e-6)
    scores = snn_counts * knn
    # final 
    scores = scores * np.exp(- (dist/10 * np.max(dist, axis = 1)[:, None]))
    scores = scores/np.sum(scores, axis = 1)[:,None]
    # Transform
    z_atac = torch.FloatTensor(scores).mm(z_rna)
    return z_rna, z_atac    


# ----------------------------------------------------- # 

# Others

# ----------------------------------------------------- # 
def assign_cluster(X, relocate_empty = False, n_relocate = 10):
    """\
    Description:
    ------------
        Select the largest element as cluster assignment
    Parameter:
    ------------
        X: (n_cells, n_clusters)
        relocate_empty: relocate empty clusters or not
        n_relocate: number of clusters for relocation
    """
    # raw cluster assignment
    clusts = np.argmax(X, axis = 1)
    empty_clusts = set([x for x in range(X.shape[1])]) - set([x for x in np.unique(clusts)])

    # find relocation, useful when finding feature clusters
    if (relocate_empty == True) and len(empty_clusts) != 0:
        flag_relocated = np.zeros_like(clusts)
        for empty_clust in empty_clusts:
            count = 0
            for reassign_idx in np.argsort(X[:,empty_clust])[::-1]:
                if flag_relocated[reassign_idx] == 0:
                    clusts[reassign_idx] = empty_clust
                    flag_relocated[reassign_idx] = 1
                    count += 1
                if count == n_relocate:
                    break
    
    # empty_clusts = set([x for x in range(X.shape[1])]) - set([x for x in np.unique(clusts)])
    # assert len(empty_clusts) == 0
    return clusts

def binarize_factor(X, relocate_empty = False, n_relocate = 10):
    """\
    Description:
    ------------
        Binarize factor matrix
    Parameter:
    ------------
        X: (n_cells, n_clusters)
        relocate_empty: relocate empty clusters or not
        n_relocate: number of clusters for relocation
    """    
    # raw cluster assignment
    bin_X = (X == np.max(X, axis = 1)[:, None])
    empty_clusts = set([x for x in range(X.shape[1])]) - set([x for x in np.unique(np.argmax(X, axis = 1))])

    # find relocation, useful when finding feature clusters
    if (relocate_empty == True) and len(empty_clusts) != 0:
        for empty_clust in empty_clusts:
            reassign_idx = np.argsort(X[:,empty_clust])[::-1]
            bin_X[reassign_idx[:n_relocate], empty_clust] = True

    return bin_X

def binarize_factor2(X, n_select = 100):
    """\
    Description:
    ------------
        Binarize factor matrix, another way is to select the max (similar to scAI)
    Parameter:
    ------------
        X: (n_cells, n_clusters)
        relocate_empty: relocate empty clusters or not
        n_relocate: number of clusters for relocation
    """
    bin_X = np.zeros_like(X).astype(np.bool)
    for clust in range(X.shape[1]):
        reassign_idx = np.argsort(X[:,clust])[::-1]
        bin_X[reassign_idx[:n_select], clust] = True
    
    # remaining, give assignment
    remain_feats = np.where(np.sum(bin_X, axis = 1) == 0)[0]
    bin_X[remain_feats,:] = (X[remain_feats,:] == np.max(X[remain_feats,:], axis = 1)[:,None])
    return bin_X

def segment1d(x):
    """\
    Description:
    ------------
        Clustering 1d array x, not just selecting the largest
    Parameter:
    ------------
        x: 1d array
    """
    
    from sklearn.neighbors import KernelDensity
    x = x.reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=np.max(x)/30).fit(x)
    s = np.linspace(0,np.max(x))
    e = kde.score_samples(s.reshape(-1,1))
    cutoff = s[np.argmin(e)]
    return cutoff

def infer_interaction(C1, C2, mask = None):
    # currently not using the association matrix
    
    # assert that the numbers of factors are the same
    assert C1.shape[1] == C2.shape[1]

    # calculate pearson correlationship
    factor_mean1 = np.mean(C1, axis = 1)[:, None]
    factor_mean2 = np.mean(C2, axis = 1)[:, None]
    var1 = np.sqrt(np.sum((C1 - factor_mean1) ** 2, axis = 1))
    var2 = np.sqrt(np.sum((C2 - factor_mean2) ** 2, axis = 1))
    cov = (C1 - factor_mean1) @ (C2 - factor_mean2).T
    p = cov/var1[:,None]/var2[None,:]

    # should be absolute value
    p = np.abs(p)

    # add the mask
    if mask is not None:
        p = mask * p
    return p


    # cannot filter correlationship using specific features
    # # select factor specific features
    # cutoff = 0.33
    # bin_C1 = np.zeros_like(C1)
    # bin_C2 = np.zeros_like(C2)
    # for factor_id in range(C1.shape[1]):
    #     # bin_C1[C1[:,factor_id] > cutoff, factor_id] = 1
    #     # bin_C2[C2[:,factor_id] > cutoff, factor_id] = 1
    #     feats1 = np.where(C1[:,factor_id] > cutoff)[0]
    #     feats2 = np.where(C2[:,factor_id] > cutoff)[0]
