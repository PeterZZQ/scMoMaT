import numpy as np
import torch
# from torch_sparse import SparseTensor
import matplotlib.pyplot as plt


import numpy as np
from scipy import stats

import torch.nn.functional as F

def lsi_ATAC(X, k = 100, use_first = False):
    """\
    Description:
    ------------
        Compute LSI with TF-IDF transform, i.e. SVD on document matrix, can do tsne on the reduced dimension

    Parameters:
    ------------
        X: cell by feature(region) count matrix
        k: number of latent dimensions
        use_first: since we know that the first LSI dimension is related to sequencing depth, we just ignore the first dimension since, and only pass the 2nd dimension and onwards for t-SNE
    
    Returns:
    -----------
        latent: cell latent matrix
    """    
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD

    # binarize the scATAC-Seq count matrix
    bin_X = np.where(X < 1, 0, 1)
    
    # perform Latent Semantic Indexing Analysis
    # get TF-IDF matrix
    tfidf = TfidfTransformer(norm='l2', sublinear_tf=True)
    normed_count = tfidf.fit_transform(bin_X)

    # perform SVD on the sparse matrix
    lsi = TruncatedSVD(n_components = k, random_state=42)
    lsi_r = lsi.fit_transform(normed_count)
    
    # use the first component or not
    if use_first:
        return lsi_r
    else:
        return lsi_r[:, 1:]

def kl_div(C):
    C_uni = torch.ones_like(C)
    C_uni = C_uni/torch.sum(C_uni, dim = 1)[:,None]
    return torch.median(torch.sum(C * torch.log(C/C_uni), dim = 1))
    


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

def preprocess(counts, mode = "standard", modality = "RNA"):
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
        # axs[0].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
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
        axs[1].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
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
        axs[0].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(0.54, 1), markerscale=4)
        
        axs[0].tick_params(axis = "both", which = "major", labelsize = 15)

        axs[0].set_xlabel(axis_label + " 1", fontsize = 19)
        axs[0].set_ylabel(axis_label + " 2", fontsize = 19)
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False) 
        
        
    if save:
        fig.savefig(save, bbox_inches = "tight")
    

# def csr2st(A):
#     A = A.tocoo()
#     col = torch.LongTensor(A.col)
#     row = torch.LongTensor(A.row)
#     value = torch.FloatTensor(A.data)
#     sparse_sizes = A.shape
#     return SparseTensor(row=row, col=col, value=value, sparse_sizes=sparse_sizes)


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

def match_alignment(z_rna, z_atac, k = 10):
    # note that the distance is squared version
    dist = _pairwise_distances(z_atac, z_rna).numpy()
    knn_index = np.argpartition(dist, kth = k - 1, axis = 1)[:,(k-1)]
    kth_dist = np.take_along_axis(dist, knn_index[:,None], axis = 1)
    
    K = dist/(kth_dist + 1e-6) 
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



###############################################################################


def compute_pairwise_distances(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def _gaussian_kernel_matrix(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sigmas = torch.FloatTensor([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]).to(device)
    dist = compute_pairwise_distances(x, y)
    beta = 1. / (2. * sigmas[:,None])
    s = - beta.mm(dist.reshape((1, -1)) )
    result =  torch.sum(torch.exp(s), dim = 0)
    return result


def maximum_mean_discrepancy(x, y): #Function to calculate MMD value
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cost = torch.mean(_gaussian_kernel_matrix(x, x))
    cost += torch.mean(_gaussian_kernel_matrix(y, y))
    cost -= 2.0 * torch.mean(_gaussian_kernel_matrix(x, y))
    cost = torch.sqrt(cost ** 2 + 1e-9)
    if cost.data.item()<0:
        cost = torch.FloatTensor([0.0]).to(device)

    return cost