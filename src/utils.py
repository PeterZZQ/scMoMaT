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
        "markerscale": 1,
    }
    _kwargs.update(kwargs)

    fig = plt.figure(figsize = figsize)
    if mode == "modality":
        colormap = plt.cm.get_cmap("Paired", len(zs))
        ax = fig.add_subplot()
        
        for batch in range(len(zs)):
            ax.scatter(zs[batch][:,0], zs[batch][:,1], color = colormap(batch), label = "batch " + str(batch), s = _kwargs["s"], alpha = _kwargs["alpha"])
        ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = _kwargs["markerscale"])
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
        cluster_types = sorted(list(cluster_types))
        for i, cluster_type in enumerate(cluster_types):
            z_clust = []
            for batch in range(len(zs)):
                index = np.where(annos[batch] == cluster_type)[0]
                z_clust.append(zs[batch][index,:])
            ax.scatter(np.concatenate(z_clust, axis = 0)[:,0], np.concatenate(z_clust, axis = 0)[:,1], color = colormap(i), label = cluster_type, s = _kwargs["s"], alpha = _kwargs["alpha"])
        
        ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = _kwargs["markerscale"])
        
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
        cluster_types = sorted(list(cluster_types))
        colormap = plt.cm.get_cmap("tab20", len(cluster_types))


        for batch in range(len(zs)):
            z_clust = []
            for i, cluster_type in enumerate(cluster_types):
                index = np.where(annos[batch] == cluster_type)[0]
                axs[batch].scatter(zs[batch][index,0], zs[batch][index,1], color = colormap(i), label = cluster_type, s = _kwargs["s"], alpha = _kwargs["alpha"])
            
            axs[batch].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(0.94, 1), markerscale = _kwargs["markerscale"])
            axs[batch].set_title("batch " + str(batch + 1), fontsize = 25)

            axs[batch].tick_params(axis = "both", which = "major", labelsize = 15)

            axs[batch].set_xlabel(axis_label + " 1", fontsize = 19)
            axs[batch].set_ylabel(axis_label + " 2", fontsize = 19)
            # axs[batch].set_xlim(np.min(np.concatenate((z1[:,0], z2[:,0]))), np.max(np.concatenate((z1[:,0], z2[:,0]))))
            # axs[batch].set_ylim(np.min(np.concatenate((z1[:,1], z2[:,1]))), np.max(np.concatenate((z1[:,1], z2[:,1]))))
            axs[batch].spines['right'].set_visible(False)
            axs[batch].spines['top'].set_visible(False)  
        
        
    if save:
        fig.savefig(save, bbox_inches = "tight")

def plot_feat_score(C_feats, n_feats = 10, figsize= (20,20), save_as = None, title = None, **kwargs):
    """\
    Description:
    ------------
        Plot feature scoring curve.
    
    Parameters:
    ------------
        C_feats:
            Dataframe of the shape (n_features, n_clusters).
        n_feats:
            Number of marked features.
    """
    
    _kwargs = {
        "s": 5,
        "alpha": 0.9,
        "fontsize": 15
    }
    _kwargs.update(kwargs)
    
    n_clusts = C_feats.shape[1]
    
    if n_clusts >= 2:
        nrows = np.ceil(n_clusts/2).astype('int32')
        ncols = 2
    elif n_clusts == 1:
        nrows = 1
        ncols = 1        
        
    if n_feats > 10:
        legend_cols = 2
    else:
        legend_cols = 1

    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
    
    if title:
        fig.suptitle("feature factor loadings", fontsize = 18)


    for clust in range(n_clusts):
        feat_factor = C_feats.iloc[:, clust].values.squeeze()
        # sort the value from the largest to the smallest
        indices = np.argsort(feat_factor)
        
        de_indices = indices[-1:-n_feats:-1]
        
        if nrows != 1:
            axs[clust%nrows, clust//nrows].scatter(np.arange(0, 1, 1/feat_factor.shape[0]), 
                                           feat_factor[indices], color = "gray", alpha = _kwargs["alpha"], s =_kwargs["s"])
            
            for i, de_idx in enumerate(de_indices):
                axs[clust%nrows, clust//nrows].scatter(np.arange(0, 1, 1/feat_factor.shape[0])[-i -1], 
                                                       feat_factor[de_idx], c = "red", s = _kwargs["s"], 
                                                       label = C_feats.index.values[de_idx])
                

            axs[clust%nrows, clust//nrows].set_title("factor " + str(clust), fontsize = 25)
            axs[clust%nrows, clust//nrows].set_ylabel("factor value", fontsize = 19)
            axs[clust%nrows, clust//nrows].set_xticks([])
            axs[clust%nrows, clust//nrows].spines['right'].set_visible(False)
            axs[clust%nrows, clust//nrows].spines['top'].set_visible(False)

            leg = axs[clust%nrows, clust//nrows].legend(bbox_to_anchor=(0.9,1), ncol = legend_cols, loc="upper left", fontsize = _kwargs["fontsize"], frameon=False)
            for item in leg.legendHandles:
                item.set_visible(False)
          
        elif nrows == 1 and ncols == 1:
           
            axs.scatter(np.arange(0, 1, 1/feat_factor.shape[0]), feat_factor[indices], color = 'gray', alpha = _kwargs["alpha"], s =_kwargs["s"])

            for i, de_idx in enumerate(de_indices):
                axs.scatter(np.arange(0, 1, 1/feat_factor.shape[0])[-i -1], 
                            feat_factor[de_idx], c = "red", s = _kwargs["s"], 
                            label = C_feats.index.values[de_idx])
                
            axs.set_title("factor " + str(clust), fontsize = 25)
            axs.set_ylabel("factor value", fontsize = 19)
            axs.set_xticks([])
            axs.spines['right'].set_visible(False)
            axs.spines['top'].set_visible(False)
            leg = axs.legend(bbox_to_anchor=(0.9,1), ncol = legend_cols, loc="upper left", fontsize = _kwargs["fontsize"], frameon=False)
            for item in leg.legendHandles:
                item.set_visible(False)

        else:
            axs[clust].scatter(np.arange(0, 1, 1/feat_factor.shape[0]), feat_factor[indices], color = 'gray', alpha = _kwargs["alpha"], s =_kwargs["s"])

            for i, de_idx in enumerate(de_indices):
                axs[clust].scatter(np.arange(0, 1, 1/feat_factor.shape[0])[-i -1], 
                                   feat_factor[de_idx], c = "red", s = _kwargs["s"], 
                                   label = C_feats.index.values[de_idx])
                
            axs[clust].set_title("factor " + str(i), fontsize = 25)
            axs[clust].set_ylabel("factor value", fontsize = 19)
            axs[clust].set_xticks([])
            axs[clust].spines['right'].set_visible(False)
            axs[clust].spines['top'].set_visible(False)
            leg = axs[clust].legend(bbox_to_anchor=(0.9,1), ncol = legend_cols, loc="upper left", fontsize = _kwargs["fontsize"], frameon=False)
            for item in leg.legendHandles:
                item.set_visible(False)
 
    fig.tight_layout(pad=0.0)
    if save_as!= None:
        fig.savefig(save_as, bbox_inches = 'tight')
    
    plt.show()     

# ----------------------------------------------------- # 

# Post-processing steps

# ----------------------------------------------------- # 
def _pairwise_distances_torch(x, y = None):
    """\
    Description:
    ------------
        Calculate pairwise distance torch version
    """
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

def _pairwise_distances(x, y = None):
    """\
    Description:
    ------------
        Calculate pairwise distance numpy version
    """
    x_norm = (x ** 2).sum(1).reshape(-1, 1)
    if y is not None:
        y_t = y.T
        y_norm = (y ** 2).sum(1).reshape(1, -1)
    else:
        y_t = x.T
        y_norm = x_norm.reshape(1, -1)
    
    dist = x_norm + y_norm - 2.0 * x @ y_t 
    if y is None:
        dist = dist - np.diag(np.diag(dist))
    
    dist[dist < 0] = 0
    return dist



def match_embeds(C_cells, k = 10, reference = None, bandwidth = 40):
    """\
    Description:
    ------------
        Matching the cell factors, multi-batches
    Parameters:
    ------------
        C_cells:
            The cell factors
        k:
            k-nearest neighbors
        reference:
            reference batch No., default None
    Return:
    ------------
        C_cells:
            The cell factors
    """
    # select reference
    if reference is None:
        batchsize = np.array([x.shape[0] for x in C_cells])
        reference = np.argmax(batchsize)
    
    C_cells_ref = C_cells[reference]

    for batch in range(len(C_cells)):

        if batch != reference:
            dist = _pairwise_distances(C_cells[batch], C_cells_ref)
            # find the index of each cell's k-th nearest neighbor
            knn_index = np.argpartition(dist, kth = k - 1, axis = 1)[:,(k-1)]
            # find the value of the k-th distance
            kth_dist = np.take_along_axis(dist, knn_index[:,None], axis = 1)

            # divide the distance by bandwidth * kth_distance, such that the K value is within the range of (0 (itself), 1/bandwidth (k-th neighbor)]
            K = dist/(bandwidth * kth_dist + 1e-6) 
            # construct knn graph with Gaussian kernel K, such that the K value is within the range of [np.exp(-1/bandwidth^2), 1 (itself)), 
            # the larger the bandwidth is, the smaller the difference is between weight of the nearest and the k-th neighbor
            K = (dist <= kth_dist) * np.exp(-K) 
            # weight sum up to 1, match batch to reference
            K = K/np.sum(K, axis = 1)[:, None]

            C_cells[batch] = K @ C_cells_ref
    
    return C_cells
        

def match_embeds_clust(C_cells, k = 10, reference = None, bandwidth = 40):
    """\
    Description:
    ------------
        Matching the cell factors, multi-batches, cluster version
    Parameters:
    ------------
        C_cells:
            The cell factors
        k:
            k-nearest neighbors
        reference:
            reference batch No., default None
    Return:
    ------------
        C_cells:
            The cell factors
    """
    # select reference
    if reference is None:
        batchsize = np.array([x.shape[0] for x in C_cells])
        reference = np.argmax(batchsize)
    
    C_cells_ref = C_cells[reference]
    clusts_ref = np.argmax(C_cells_ref, axis = 1).squeeze()

    for batch in range(len(C_cells)):
        if batch != reference:

            # find cells correspond to the cluster
            clusts_batch = np.argmax(C_cells[batch], axis = 1).squeeze()
            for clust in range(C_cells_ref.shape[1]):
                clust_idx = np.where(clusts_batch == clust)[0]
                clust_ref_idx = np.where(clusts_ref == clust)[0]                
                
                if (clust_idx.shape[0] != 0) and (clust_ref_idx.shape[0] != 0):
                    C_clust_ref = C_cells_ref[clust_ref_idx, :]
                    C_clust = C_cells[batch][clust_idx, :]     

                elif clust_idx.shape[0] != 0: # if the reference cluster does not exist
                    # uses all the reference data
                    C_clust_ref = C_cells_ref
                    C_clust = C_cells[batch][clust_idx, :]
                
                else: # if the batch cluster does not exist, skip
                    continue

                dist = _pairwise_distances(C_clust, C_clust_ref)
                # find the index of each cell's k-th nearest neighbor
                knn_index = np.argpartition(dist, kth = k - 1, axis = 1)[:,(k-1)]
                # find the value of the k-th distance
                kth_dist = np.take_along_axis(dist, knn_index[:,None], axis = 1)

                # divide the distance by bandwidth * kth_distance, such that the K value is within the range of (0 (itself), 1/bandwidth (k-th neighbor)]
                K = dist/(bandwidth * kth_dist + 1e-6) 
                # construct knn graph with Gaussian kernel K, such that the K value is within the range of [np.exp(-1/bandwidth^2), 1 (itself)), 
                # the larger the bandwidth is, the smaller the difference is between weight of the nearest and the k-th neighbor
                K = (dist <= kth_dist) * np.exp(-K) 
                # weight sum up to 1, match batch to reference
                K = K/np.sum(K, axis = 1)[:, None]

                C_cells[batch][clust_idx, :] = K @ C_clust_ref

    
    return C_cells


# TODO: check Seurat version, implement graphical UMAP for post-processing
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


def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig
    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except:
        pass
    if g.vcount() != adjacency.shape[0]:
        print( 'Your adjacency matrix contained redundant nodes.' )
    return g


def leiden_cluster(
    X, 
    resolution = 30.0,
    random_state = 0,
    n_iterations: int = -1,
    k_neighs = 30,
    sigma = 1,
    **partition_kwargs):

    from sklearn.neighbors import NearestNeighbors

    try:
        import leidenalg
    except ImportError:
        raise ImportError(
            'Please install the leiden algorithm: `conda install -c conda-forge leidenalg` or `pip3 install leidenalg`.'
        )

    partition_kwargs = dict(partition_kwargs)


    neighbor = NearestNeighbors(n_neighbors = k_neighs)
    neighbor.fit(X)
    # get test connectivity result 0-1 adj_matrix, mode = 'connectivity' by default
    adj_matrix = neighbor.kneighbors_graph(X).toarray()
    dist_matrix = neighbor.kneighbors_graph(X, mode='distance').toarray()

    adj_matrix += adj_matrix.T
    # change 2 back to 1
    adj_matrix[adj_matrix.nonzero()[0],adj_matrix.nonzero()[1]] = 1

    affin = np.exp(- (dist_matrix - np.min(dist_matrix, axis = 1)[:,None]) ** 2/sigma)
    affin = adj_matrix * affin
    affin = (affin + affin.T)/2
        
    partition_type = leidenalg.RBConfigurationVertexPartition
    g = get_igraph_from_adjacency(affin, directed = True)

    partition_kwargs['n_iterations'] = n_iterations
    partition_kwargs['seed'] = random_state
    partition_kwargs['resolution_parameter'] = resolution

    partition_kwargs['weights'] = np.array(g.es['weight']).astype(np.float64)
    part = leidenalg.find_partition(g, partition_type, **partition_kwargs)
    groups = np.array(part.membership)

    return groups



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


'''
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

'''



# def umap(
#     adata: AnnData,
#     min_dist: float = 0.5,
#     spread: float = 1.0,
#     n_components: int = 2,
#     maxiter: Optional[int] = None,
#     alpha: float = 1.0,
#     gamma: float = 1.0,
#     negative_sample_rate: int = 5,
#     init_pos: Union[_InitPos, np.ndarray, None] = 'spectral',
#     random_state: AnyRandom = 0,
#     a: Optional[float] = None,
#     b: Optional[float] = None,
#     copy: bool = False,
#     method: Literal['umap', 'rapids'] = 'umap',
#     neighbors_key: Optional[str] = None,
# ) -> Optional[AnnData]:
#     """\
#     Embed the neighborhood graph using UMAP [McInnes18]_.
#     UMAP (Uniform Manifold Approximation and Projection) is a manifold learning
#     technique suitable for visualizing high-dimensional data. Besides tending to
#     be faster than tSNE, it optimizes the embedding such that it best reflects
#     the topology of the data, which we represent throughout Scanpy using a
#     neighborhood graph. tSNE, by contrast, optimizes the distribution of
#     nearest-neighbor distances in the embedding such that these best match the
#     distribution of distances in the high-dimensional space.  We use the
#     implementation of `umap-learn <https://github.com/lmcinnes/umap>`__
#     [McInnes18]_. For a few comparisons of UMAP with tSNE, see this `preprint
#     <https://doi.org/10.1101/298430>`__.
#     Parameters
#     ----------
#     adata
#         Annotated data matrix.
#     min_dist
#         The effective minimum distance between embedded points. Smaller values
#         will result in a more clustered/clumped embedding where nearby points on
#         the manifold are drawn closer together, while larger values will result
#         on a more even dispersal of points. The value should be set relative to
#         the ``spread`` value, which determines the scale at which embedded
#         points will be spread out. The default of in the `umap-learn` package is
#         0.1.
#     spread
#         The effective scale of embedded points. In combination with `min_dist`
#         this determines how clustered/clumped the embedded points are.
#     n_components
#         The number of dimensions of the embedding.
#     maxiter
#         The number of iterations (epochs) of the optimization. Called `n_epochs`
#         in the original UMAP.
#     alpha
#         The initial learning rate for the embedding optimization.
#     gamma
#         Weighting applied to negative samples in low dimensional embedding
#         optimization. Values higher than one will result in greater weight
#         being given to negative samples.
#     negative_sample_rate
#         The number of negative edge/1-simplex samples to use per positive
#         edge/1-simplex sample in optimizing the low dimensional embedding.
#     init_pos
#         How to initialize the low dimensional embedding. Called `init` in the
#         original UMAP. Options are:
#         * Any key for `adata.obsm`.
#         * 'paga': positions from :func:`~scanpy.pl.paga`.
#         * 'spectral': use a spectral embedding of the graph.
#         * 'random': assign initial embedding positions at random.
#         * A numpy array of initial embedding positions.
#     random_state
#         If `int`, `random_state` is the seed used by the random number generator;
#         If `RandomState` or `Generator`, `random_state` is the random number generator;
#         If `None`, the random number generator is the `RandomState` instance used
#         by `np.random`.
#     a
#         More specific parameters controlling the embedding. If `None` these
#         values are set automatically as determined by `min_dist` and
#         `spread`.
#     b
#         More specific parameters controlling the embedding. If `None` these
#         values are set automatically as determined by `min_dist` and
#         `spread`.
#     copy
#         Return a copy instead of writing to adata.
#     method
#         Use the original 'umap' implementation, or 'rapids' (experimental, GPU only)
#     neighbors_key
#         If not specified, umap looks .uns['neighbors'] for neighbors settings
#         and .obsp['connectivities'] for connectivities
#         (default storage places for pp.neighbors).
#         If specified, umap looks .uns[neighbors_key] for neighbors settings and
#         .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.
#     Returns
#     -------
#     Depending on `copy`, returns or updates `adata` with the following fields.
#     **X_umap** : `adata.obsm` field
#         UMAP coordinates of data.
#     """
#     adata = adata.copy() if copy else adata

#     if neighbors_key is None:
#         neighbors_key = 'neighbors'

#     if neighbors_key not in adata.uns:
#         raise ValueError(
#             f'Did not find .uns["{neighbors_key}"]. Run `sc.pp.neighbors` first.'
#         )
#     start = logg.info('computing UMAP')

#     neighbors = NeighborsView(adata, neighbors_key)

#     if 'params' not in neighbors or neighbors['params']['method'] != 'umap':
#         logg.warning(
#             f'.obsp["{neighbors["connectivities_key"]}"] have not been computed using umap'
#         )

#     # Compat for umap 0.4 -> 0.5
#     with warnings.catch_warnings():
#         # umap 0.5.0
#         warnings.filterwarnings("ignore", message=r"Tensorflow not installed")
#         import umap

#     if version.parse(umap.__version__) >= version.parse("0.5.0"):

#         def simplicial_set_embedding(*args, **kwargs):
#             from umap.umap_ import simplicial_set_embedding

#             X_umap, _ = simplicial_set_embedding(
#                 *args,
#                 densmap=False,
#                 densmap_kwds={},
#                 output_dens=False,
#                 **kwargs,
#             )
#             return X_umap

#     else:
#         from umap.umap_ import simplicial_set_embedding
#     from umap.umap_ import find_ab_params

#     if a is None or b is None:
#         a, b = find_ab_params(spread, min_dist)
#     else:
#         a = a
#         b = b
#     adata.uns['umap'] = {'params': {'a': a, 'b': b}}
#     if isinstance(init_pos, str) and init_pos in adata.obsm.keys():
#         init_coords = adata.obsm[init_pos]
#     elif isinstance(init_pos, str) and init_pos == 'paga':
#         init_coords = get_init_pos_from_paga(
#             adata, random_state=random_state, neighbors_key=neighbors_key
#         )
#     else:
#         init_coords = init_pos  # Let umap handle it
#     if hasattr(init_coords, "dtype"):
#         init_coords = check_array(init_coords, dtype=np.float32, accept_sparse=False)

#     if random_state != 0:
#         adata.uns['umap']['params']['random_state'] = random_state
#     random_state = check_random_state(random_state)

#     neigh_params = neighbors['params']
#     X = _choose_representation(
#         adata,
#         neigh_params.get('use_rep', None),
#         neigh_params.get('n_pcs', None),
#         silent=True,
#     )
#     if method == 'umap':
#         # the data matrix X is really only used for determining the number of connected components
#         # for the init condition in the UMAP embedding
#         n_epochs = 0 if maxiter is None else maxiter
#         X_umap = simplicial_set_embedding(
#             X,
#             neighbors['connectivities'].tocoo(),
#             n_components,
#             alpha,
#             a,
#             b,
#             gamma,
#             negative_sample_rate,
#             n_epochs,
#             init_coords,
#             random_state,
#             neigh_params.get('metric', 'euclidean'),
#             neigh_params.get('metric_kwds', {}),
#             verbose=settings.verbosity > 3,
#         )

#     adata.obsm['X_umap'] = X_umap  # annotate samples with UMAP coordinates
#     logg.info(
#         '    finished',
#         time=start,
#         deep=('added\n' "    'X_umap', UMAP coordinates (adata.obsm)"),
#     )
#     return adata if copy else None