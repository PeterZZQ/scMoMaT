from signal import raise_signal
import numpy as np
from numpy import block
import torch
# from torch_sparse import SparseTensor
import matplotlib.pyplot as plt


import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import scipy.sparse as sp
from umap.utils import fast_knn_indices

from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text
import time
from multiprocessing import Pool, cpu_count

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

def quantile_norm_log(X, log = True):
    if log:
        logX = np.log1p(X)
    else:
        logX = X
    logXn = quantile_norm(logX)
    return logXn

def _preprocess_old(counts, mode = "standard", modality = "RNA"):
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

def preprocess(counts, modality = "RNA", log = True):
    """\
    Description:
    ------------
    Preprocess the dataset, for count, interaction matrices
    """
    if modality == "ATAC":
        # make binary, maximum is 1
        counts = (counts > 0).astype(np.float) 
        # # normalize according to library size
        # counts = counts / np.sum(counts, axis = 1)[:,None]
        # counts = counts/np.max(counts)

    elif modality == "interaction":
        # gene by region matrix
        counts = counts/(np.sum(counts, axis = 1)[:,None] + 1e-6)
    
    else:
        # other cases, e.g. Protein, RNA, etc
        counts = quantile_norm_log(counts, log = log)
        counts = counts/np.max(counts)

    return counts

def preprocess_liger(counts, with_mean = False):
    # normalize the count (library size) of the data
    counts = counts/(np.sum(counts, axis = 1, keepdims = True) + 1e-6)
    # scale for unit variance, 
    # vars stores the variance of each feature
    # vars = np.sum(counts ** 2, axis = 0, keepdims = True)
    counts = StandardScaler(with_mean = with_mean, with_std = True).fit_transform(counts)
    return counts


# ----------------------------------------------------- # 

# Plot

# ----------------------------------------------------- # 

def plot_latent_ext(zs, annos = None, mode = "joint", save = None, figsize = (20,10), axis_label = "Latent", label_inplace = False, **kwargs):
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
        "text_size": "xx-large",
        "colormap": "tab20b"
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
        colormap = plt.cm.get_cmap(_kwargs["colormap"], len(cluster_types))
        cluster_types = sorted(list(cluster_types))
        
        texts = []
        for i, cluster_type in enumerate(cluster_types):
            z_clust = []
            for batch in range(len(zs)):
                index = np.where(annos[batch] == cluster_type)[0]
                z_clust.append(zs[batch][index,:])
            ax.scatter(np.concatenate(z_clust, axis = 0)[:,0], np.concatenate(z_clust, axis = 0)[:,1], color = colormap(i), label = cluster_type, s = _kwargs["s"], alpha = _kwargs["alpha"])
            # text on plot
            if label_inplace:
                texts.append(ax.text(np.median(np.concatenate(z_clust, axis = 0)[:,0]), np.median(np.concatenate(z_clust, axis = 0)[:,1]), color = "black", s = cluster_types[i], fontsize = _kwargs["text_size"], weight = 'semibold', in_layout = True))
        
        ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = (len(cluster_types) // 15) + 1, bbox_to_anchor=(1.04, 1), markerscale = _kwargs["markerscale"])
        
        ax.tick_params(axis = "both", which = "major", labelsize = 15)

        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  
        # adjust position
        if label_inplace:
            adjust_text(texts, only_move={'points':'xy', 'texts':'xy'})

    elif mode == "separate":
        axs = fig.subplots(len(zs),1)
        cluster_types = set()
        for batch in range(len(zs)):
            cluster_types = cluster_types.union(set([x for x in np.unique(annos[batch])]))
        cluster_types = sorted(list(cluster_types))
        colormap = plt.cm.get_cmap(_kwargs["colormap"], len(cluster_types))


        for batch in range(len(zs)):
            z_clust = []
            texts = []
            for i, cluster_type in enumerate(cluster_types):
                index = np.where(annos[batch] == cluster_type)[0]
                axs[batch].scatter(zs[batch][index,0], zs[batch][index,1], color = colormap(i), label = cluster_type, s = _kwargs["s"], alpha = _kwargs["alpha"])
                # text on plot
                if label_inplace:
                    texts.append(axs[batch].text(np.median(zs[batch][index,0]), np.median(zs[batch][index,1]), color = "black", s = cluster_types[i], fontsize = _kwargs["text_size"], weight = 'semibold', in_layout = True))
            
            axs[batch].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = (len(cluster_types) // 15) + 1, bbox_to_anchor=(0.94, 1), markerscale = _kwargs["markerscale"])
            axs[batch].set_title("batch " + str(batch + 1), fontsize = 25)

            axs[batch].tick_params(axis = "both", which = "major", labelsize = 15)

            axs[batch].set_xlabel(axis_label + " 1", fontsize = 19)
            axs[batch].set_ylabel(axis_label + " 2", fontsize = 19)
            # axs[batch].set_xlim(np.min(np.concatenate((z1[:,0], z2[:,0]))), np.max(np.concatenate((z1[:,0], z2[:,0]))))
            # axs[batch].set_ylim(np.min(np.concatenate((z1[:,1], z2[:,1]))), np.max(np.concatenate((z1[:,1], z2[:,1]))))
            axs[batch].spines['right'].set_visible(False)
            axs[batch].spines['top'].set_visible(False)  
            if label_inplace:
                adjust_text(texts, only_move={'points':'xy', 'texts':'xy'})        
    plt.tight_layout()
    if save:
        fig.savefig(save, bbox_inches = "tight")

def plot_latent_continuous(zs, annos = None, mode = "joint", save = None, title = None, figsize = (20,10), axis_label = "Latent", **kwargs):
    """\
    Description
        Plot latent space with continuous values
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
        "text_size": "xx-large",
        "cmap": "gnuplot"
    }
    _kwargs.update(kwargs)

    fig = plt.figure(figsize = figsize)
    if mode == "joint":
        ax = fig.add_subplot()
        

        p = ax.scatter(np.concatenate(zs, axis = 0)[:,0], np.concatenate(zs, axis = 0)[:,1], c = np.concatenate(annos), cmap=plt.get_cmap(_kwargs["cmap"]), s = _kwargs["s"], alpha = _kwargs["alpha"])
                
        ax.tick_params(axis = "both", which = "major", labelsize = 15)
        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if title is not None:
            ax.set_title(title)
        cbar = fig.colorbar(p, fraction=0.046, pad=0.04, ax = ax)
        cbar.ax.tick_params(labelsize = 20)


    elif mode == "separate":
        axs = fig.subplots(len(zs),1)

        for batch in range(len(zs)):
            p = axs[batch].scatter(zs[batch][:,0], zs[batch][:,1], c = annos[batch], cmap=plt.get_cmap(_kwargs["cmap"]), s = _kwargs["s"], alpha = _kwargs["alpha"])
            
            axs[batch].set_title("batch " + str(batch + 1), fontsize = 25)
            axs[batch].tick_params(axis = "both", which = "major", labelsize = 15)

            axs[batch].set_xlabel(axis_label + " 1", fontsize = 19)
            axs[batch].set_ylabel(axis_label + " 2", fontsize = 19)
            # axs[batch].set_xlim(np.min(np.concatenate((z1[:,0], z2[:,0]))), np.max(np.concatenate((z1[:,0], z2[:,0]))))
            # axs[batch].set_ylim(np.min(np.concatenate((z1[:,1], z2[:,1]))), np.max(np.concatenate((z1[:,1], z2[:,1]))))
            axs[batch].spines['right'].set_visible(False)
            axs[batch].spines['top'].set_visible(False)  
            
            cbar = fig.colorbar(p, fraction=0.046, pad=0.04, ax = axs[batch])
            cbar.ax.tick_params(labelsize = 20)

    plt.tight_layout()
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
                

            axs[clust%nrows, clust//nrows].set_title("cluster " + str(clust), fontsize = 25)
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
                
            axs.set_title("cluster " + str(clust), fontsize = 25)
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
                
            axs[clust].set_title("cluster " + str(i), fontsize = 25)
            axs[clust].set_ylabel("factor value", fontsize = 19)
            axs[clust].set_xticks([])
            axs[clust].spines['right'].set_visible(False)
            axs[clust].spines['top'].set_visible(False)
            leg = axs[clust].legend(bbox_to_anchor=(0.9,1), ncol = legend_cols, loc="upper left", fontsize = _kwargs["fontsize"], frameon=False)
            for item in leg.legendHandles:
                item.set_visible(False)
 
    fig.tight_layout(pad=3.0)
    if save_as!= None:
        fig.savefig(save_as, bbox_inches = 'tight')
    
    plt.show()     

def plot_factor(C_feats, markers, cluster = 0, figsize = (10,20)): 
    n_markers = len(markers)
    if n_markers >= 2:
        nrows = np.ceil(n_markers/2).astype('int32')
        ncols = 2
    elif n_markers == 1:
        nrows = 1
        ncols = 1 

    clusts = np.array([eval(x.split("_")[1]) for x in C_feats.columns.values.squeeze()])   
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
    for marker in range(n_markers):
        x = C_feats.loc[markers[marker],:].values.squeeze()
        if nrows != 1:
            barlist = axs[marker%nrows, marker//nrows].bar(np.arange(C_feats.shape[1]), x)
            if isinstance(cluster, list):
                for i in cluster:
                    barlist[np.where(clusts == i)[0][0]].set_color('r')    
            else:
                barlist[np.where(clusts == cluster)[0][0]].set_color('r')

            axs[marker%nrows, marker//nrows].tick_params(axis='x', labelsize=15)
            axs[marker%nrows, marker//nrows].tick_params(axis='y', labelsize=15)
            axs[marker%nrows, marker//nrows].set_title(markers[marker], fontsize = 20)
            _ = axs[marker%nrows, marker//nrows].set_xticks(np.arange(C_feats.shape[1]))
            _ = axs[marker%nrows, marker//nrows].set_xticklabels(clusts)
            _ = axs[marker%nrows, marker//nrows].set_xlabel("cluster", fontsize = 20)
            _ = axs[marker%nrows, marker//nrows].set_ylabel("factor value", fontsize = 20)
        
        elif nrows == 1 and ncols == 1:
            barlist = axs.bar(np.arange(C_feats.shape[1]), x)
            if isinstance(cluster, list):
                for i in cluster:
                    barlist[np.where(clusts == i)[0][0]].set_color('r')    
            else:
                barlist[np.where(clusts == cluster)[0][0]].set_color('r')

            axs.tick_params(axis='x', labelsize=15)
            axs.tick_params(axis='y', labelsize=15)
            axs.set_title(markers[marker], fontsize = 20)
            _ = axs.set_xticks(np.arange(C_feats.shape[1]))
            _ = axs.set_xticklabels(clusts)
            _ = axs.set_xlabel("cluster", fontsize = 20)
            _ = axs.set_ylabel("factor value", fontsize = 20)
        
        else:
            barlist = axs[marker].bar(np.arange(C_feats.shape[1]), x)
            if isinstance(cluster, list):
                for i in cluster:
                    barlist[np.where(clusts == i)[0][0]].set_color('r')       
            else:
                barlist[np.where(clusts == cluster)[0][0]].set_color('r')

            axs[marker].tick_params(axis='x', labelsize=15)
            axs[marker].tick_params(axis='y', labelsize=15)
            axs[marker].set_title(markers[marker], fontsize = 20)
            _ = axs[marker].set_xticks(np.arange(C_feats.shape[1]))
            _ = axs[marker].set_xticklabels(clusts)
            _ = axs[marker].set_xlabel("cluster", fontsize = 20)
            _ = axs[marker].set_ylabel("factor value", fontsize = 20)
    plt.tight_layout()
    return fig

# ----------------------------------------------------- # 

# Post-processing steps

# ----------------------------------------------------- # 

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


def _compute_connectivities_umap(
    knn_indices,
    knn_dists,
    n_neighbors,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
):
    """\
    This is from umap.fuzzy_simplicial_set [McInnes18]_.
    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """

    from umap.umap_ import fuzzy_simplicial_set
    from scipy.sparse import coo_matrix

    # place holder since we use precompute matrix
    X = coo_matrix(([], ([], [])), shape=(knn_indices.shape[0], 1))
    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )

    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]

    return connectivities.tocsr()

def leiden_cluster(
    X = None, 
    knn_indices = None,
    knn_dists = None,
    resolution = 30.0,
    random_state = 0,
    n_iterations: int = -1,
    k_neighs = 30,
    sigma = 1,
    affin = None,
    **partition_kwargs):

    from sklearn.neighbors import NearestNeighbors

    try:
        import leidenalg
    except ImportError:
        raise ImportError(
            'Please install the leiden algorithm: `conda install -c conda-forge leidenalg` or `pip3 install leidenalg`.'
        )

    partition_kwargs = dict(partition_kwargs)
    
    if affin is None:
        if (knn_indices is None) or (knn_dists is None):
            # X is needed
            if X is None:
                raise ValueError("`X' and `knn_indices & knn_dists', at least one need to be provided.")

            neighbor = NearestNeighbors(n_neighbors = k_neighs)
            neighbor.fit(X)
            # get test connectivity result 0-1 adj_matrix, mode = 'connectivity' by default
            knn_dists, knn_indices = neighbor.kneighbors(X, n_neighbors = k_neighs, return_distance = True)

        affin = _compute_connectivities_umap(knn_indices = knn_indices, knn_dists = knn_dists, n_neighbors = k_neighs, set_op_mix_ratio=1.0, local_connectivity=1.0)
        affin = affin.todense()
        
    partition_type = leidenalg.RBConfigurationVertexPartition
    g = get_igraph_from_adjacency(affin, directed = True)

    partition_kwargs['n_iterations'] = n_iterations
    partition_kwargs['seed'] = random_state
    partition_kwargs['resolution_parameter'] = resolution

    partition_kwargs['weights'] = np.array(g.es['weight']).astype(np.float64)
    part = leidenalg.find_partition(g, partition_type, **partition_kwargs)
    groups = np.array(part.membership)

    return groups


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





def post_process(X, n_neighbors, njobs = None):
    # get a pairwise distance matrix for all batches
    from sklearn.metrics import pairwise_distances
    start_time = time.time()
    pair_dist = pairwise_distances(np.concatenate(X, axis = 0), n_jobs = njobs)
    end_time = time.time()
    print("calculating pairwise distance, time used {:.4f}s".format(end_time - start_time))

    # start_time = time.time()
    # pair_dist2 = squareform(pdist(np.concatenate(X, axis=0)))
    # end_time = time.time()
    # print("calculating pairwise distance, time used {:.4f}s".format(end_time - start_time))
    # assert np.allclose(pair_dist, pair_dist2)
    if n_neighbors < len(X):
        raise ValueError("the number of neighbors should not be smaller than the number of batches")

    # get the start points, end points and number of neighbors each batch
    start_time = time.time()
    start_point, end_point, b_ratios = [], [], []
    start = 0

    for batch in range(len(X)):
        start_point.append(start)
        b_ratios.append(len(X[batch])/len(pair_dist)) 
        start += len(X[batch])
        end_point.append(start-1)

    # compute the number of nearest neighbors for each sample in each batch of X
    # If the number of neighbors is too small, the it might be that some batches don't have neighbors, for both sampling and fixed
    # b_neighbors = np.random.multinomial(n_neighbors, b_ratios)
    # assign b_neighbors directly according to proportion.
    b_neighbors = []
    for batch in range(len(X)):
        if batch == len(X) - 1:
            b_neighbors.append(max(int(n_neighbors - sum(b_neighbors)), 1))
        else:
            b_neighbors.append(max(int(n_neighbors * b_ratios[batch]), 1))

    # compute knn_indices based on b_neighbors
    knn_indices = np.zeros((len(pair_dist), n_neighbors))
    knn_dists = np.zeros((len(pair_dist), n_neighbors))
    for batch in range(len(X)):
        knn_indices[:, sum(b_neighbors[0:batch]):sum(b_neighbors[0:(batch+1)])] = fast_knn_indices(pair_dist[:, start_point[batch]:end_point[batch]+1], b_neighbors[batch]) + start_point[batch]
    knn_indices = knn_indices.astype(int)
    knn_dists = pair_dist[np.arange(pair_dist.shape[0])[:, None], knn_indices].copy()
    end_time = time.time()
    print("knn separate, time used {:.4f}s".format(end_time - start_time))
    
    start_time = time.time()
    for batch_i in range(len(X)):
        ref_block = knn_dists[start_point[batch_i]:(end_point[batch_i] + 1), sum(b_neighbors[0:batch_i]):sum(b_neighbors[0:(batch_i+1)])]
        for batch_j in range(len(X)):
            if batch_i != batch_j:
                block = knn_dists[start_point[batch_i]:(end_point[batch_i] + 1), sum(b_neighbors[0:batch_j]):sum(b_neighbors[0:(batch_j+1)])]
                knn_dists[start_point[batch_i]:(end_point[batch_i] + 1), sum(b_neighbors[0:batch_j]):sum(b_neighbors[0:(batch_j+1)])] = block/np.mean(block, axis = 1, keepdims = True) * np.mean(ref_block, axis = 1, keepdims = True)
    end_time = time.time()
    print("modify distance 1, time used {:.4f}s".format(end_time - start_time))

    start_time = time.time()
    # Modify pairwise distance matrix where the elements are changed due to knn_dists, 
    # pairwise_distance does not affect the UMAP visualization and leiden clustering

    # OLD
    # pair_dist[np.arange(pair_dist.shape[0])[:, None], knn_indices] = knn_dists2
    # #Ensure pairwise distance matrix is symmetric
    # pair_dist[knn_indices.T, np.arange(pair_dist.shape[1])[None, :]] = knn_dists2.T
    # # maker sure that diagonal is 0
    # pair_dist = np.triu(pair_dist, 1)
    # pair_dist += pair_dist.T

    # NEW, UMAP
    pairwise_distances = np.zeros_like(pair_dist)
    pairwise_distances[np.arange(pairwise_distances.shape[0])[:, None], knn_indices] = knn_dists
    pairwise_distances = pairwise_distances + pairwise_distances.T - pairwise_distances * pairwise_distances.T

    end_time = time.time()
    print("modify distance 2, time used {:.4f}s".format(end_time - start_time))

    # start_time = time.time()     
    pairwise_distances = sp.csr_matrix(pairwise_distances)
    # end_time = time.time()
    # print("make sparse, time used {:.4f}s".format(end_time - start_time))
    # pairwise_distances = pair_dist
    return pairwise_distances, knn_indices, knn_dists




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


'''
def _quantile_align(Xs):
    if len(Xs) == 2:
        X_query = Xs[0]
        val_ref = Xs[1]
        val_query = X_query.reshape(-1)
        assert len(val_ref) >= len(val_query)
        d = np.argsort(val_query)
        val_query[d] = np.sort(val_ref)[:len(val_query)]
        X_query = val_query.reshape(X_query.shape[0], X_query.shape[1])
        return X_query
    else:
        return Xs[0]

def post_distance_nn(X, n_neighbors, njobs = None, seed = 0):
    np.random.seed(seed)
    # get a pairwise distance matrix for all batches
    from sklearn.metrics import pairwise_distances
    start_time = time.time()
    pair_dist = pairwise_distances(np.concatenate(X, axis = 0), n_jobs = njobs)
    end_time = time.time()
    print("calculating pairwise distance, time used {:.4f}s".format(end_time - start_time))

    if n_neighbors < len(X):
        raise ValueError("the number of neighbors should not be smaller than the number of batches")

    start_time = time.time()
    # get the matrix who has the largest numbers of elements as the reference matrix
    start_point, end_point, b_ratios = [], [], []
    maxnum, maxbatch, start, number = 0, 0, 0, 0

    for batch in range(len(X)):
        start_point.append(start)
        b_ratios.append(len(X[batch])/len(pair_dist)) 
        number = len(X[batch])
        start += number
        end = start-1
        end_point.append(end)
        if number > maxnum:
            maxnum = start
            maxbatch = batch

    # pick the largest matrix as the reference matrix
    ref_dis = pair_dist[start_point[maxbatch]:(end_point[maxbatch]+1), start_point[maxbatch]:(end_point[maxbatch]+1)].reshape(-1)

    # compute the number of nearest neighbors for each sample in each batch of X
    # b_neighbors = np.random.multinomial(n_neighbors, b_ratios)
    # assign b_neighbors directly according to proportion.
    b_neighbors = []
    for batch in range(len(X)):
        if batch == len(X) - 1:
            b_neighbors.append(max(int(n_neighbors - sum(b_neighbors)), 1))
        else:
            b_neighbors.append(max(int(n_neighbors * b_ratios[batch]), 1))


    # # Modify distances for each block
    # distance = np.zeros((len(pair_dist), len(pair_dist)), dtype=np.float32)
    # for batch_i in range(len(X)):
    #     for batch_j in range(batch_i, len(X)):
    #         if [batch_i, batch_j] != [maxbatch, maxbatch]:
    #             blocks = pair_dist[start_point[batch_i]:(end_point[batch_i]+1), start_point[batch_j]:(end_point[batch_j]+1)].reshape(-1)
    #             assert len(blocks) <= len(ref_dis)
    #             d = np.argsort(blocks)

    #             sample = np.random.choice(ref_dis, len(d), replace=False)

    #             blocks[d] = np.sort(sample)
    #             distance[start_point[batch_i]:(end_point[batch_i]+1), start_point[batch_j]:(end_point[batch_j]+1)] = \
    #                 np.reshape(blocks, (end_point[batch_i]+1-start_point[batch_i], end_point[batch_j]+1-start_point[batch_j]))
    #         else:
    #             distance[start_point[batch_i]:(end_point[batch_i]+1), start_point[batch_j]:(end_point[batch_j]+1)] = \
    #                 pair_dist[start_point[batch_i]:(end_point[batch_i]+1), start_point[batch_j]:(end_point[batch_j]+1)]

    # parallel
    Blocks = []
    for batch_i in range(len(X)):
        for batch_j in range(batch_i, len(X)):
            if [batch_i, batch_j] != [maxbatch, maxbatch]:
                Blocks.append([pair_dist[start_point[batch_i]:(end_point[batch_i]+1), start_point[batch_j]:(end_point[batch_j]+1)], ref_dis])

            else:
                Blocks.append([pair_dist[start_point[batch_i]:(end_point[batch_i]+1), start_point[batch_j]:(end_point[batch_j]+1)]])

    pool = Pool(min(len(X)**2, njobs))
    Blocks_quant = pool.map(_quantile_align, [x for x in Blocks])
    pool.close()
    pool.join()
    distance = np.zeros((len(pair_dist), len(pair_dist)), dtype=np.float32)
    count = 0
    for batch_i in range(len(X)):
        for batch_j in range(batch_i, len(X)):
            distance[start_point[batch_i]:(end_point[batch_i]+1), start_point[batch_j]:(end_point[batch_j]+1)] = Blocks_quant[count]
            count += 1

    distance = np.triu(distance, 1)
    distance += distance.T
    pairwise_distances = sp.csr_matrix(distance)
    end_time = time.time()
    print("Modified pairwise distance with quantile normalization, time used {:.4f}".format(end_time - start_time))

    
    # compute knn_indices and knn_dists based on modified pairwise distances and 
    # customized number of nearest neighbors
    start_time = time.time()
    knn_indices = np.zeros((len(pair_dist), n_neighbors))
    for batch in range(len(X)):
        knn_indices[:, sum(b_neighbors[0:batch]):sum(b_neighbors[0:batch+1])] = \
                    fast_knn_indices(pair_dist[:, start_point[batch]:end_point[batch]+1], b_neighbors[batch]) \
                    + start_point[batch]
    
    knn_indices = knn_indices.astype(int)
    knn_dists = pair_dist[np.arange(pair_dist.shape[0])[:, None], knn_indices].copy()
    end_time = time.time()
    print("Calculate KNN, time used {:.4f}".format(end_time - start_time))
    
    return pairwise_distances, knn_indices, knn_dists


def post_nn_distance(X, n_neighbors, njobs = None):
    # get a pairwise distance matrix for all batches
    from sklearn.metrics import pairwise_distances
    start_time = time.time()
    pair_dist = pairwise_distances(np.concatenate(X, axis = 0), n_jobs = njobs)
    end_time = time.time()
    print("calculating pairwise distance, time used {:.4f}s".format(end_time - start_time))

    # start_time = time.time()
    # pair_dist2 = squareform(pdist(np.concatenate(X, axis=0)))
    # end_time = time.time()
    # print("calculating pairwise distance, time used {:.4f}s".format(end_time - start_time))
    # assert np.allclose(pair_dist, pair_dist2)
    if n_neighbors < len(X):
        raise ValueError("the number of neighbors should not be smaller than the number of batches")

    # get the start points, end points and number of neighbors each batch
    start_time = time.time()
    start_point, end_point, b_ratios = [], [], []
    start = 0

    for batch in range(len(X)):
        start_point.append(start)
        b_ratios.append(len(X[batch])/len(pair_dist)) 
        start += len(X[batch])
        end_point.append(start-1)

    # compute the number of nearest neighbors for each sample in each batch of X
    # If the number of neighbors is too small, the it might be that some batches don't have neighbors, for both sampling and fixed
    # b_neighbors = np.random.multinomial(n_neighbors, b_ratios)
    # assign b_neighbors directly according to proportion.
    b_neighbors = []
    for batch in range(len(X)):
        if batch == len(X) - 1:
            b_neighbors.append(max(int(n_neighbors - sum(b_neighbors)), 1))
        else:
            b_neighbors.append(max(int(n_neighbors * b_ratios[batch]), 1))

    # compute knn_indices based on b_neighbors
    knn_indices = np.zeros((len(pair_dist), n_neighbors))
    for batch in range(len(X)):
        knn_indices[:, sum(b_neighbors[0:batch]):sum(b_neighbors[0:(batch+1)])] = fast_knn_indices(pair_dist[:, start_point[batch]:end_point[batch]+1], b_neighbors[batch]) + start_point[batch]
    knn_indices = knn_indices.astype(int)
    end_time = time.time()
    print("knn separate, time used {:.4f}s".format(end_time - start_time))
    
    # Only modify distance for points in knn_indices
    # Get knn_distance from original knn_indices, which is nearest neighbors totally 
    # based on n_neighbors
    start_time = time.time() 
    ori_indices = fast_knn_indices(pair_dist, n_neighbors)
    knn_dists = pair_dist[np.arange(pair_dist.shape[0])[:, None], ori_indices].copy()
    end_time = time.time()
    print("knn joint, time used {:.4f}s".format(end_time - start_time))
    start_time = time.time() 

    # the knn_dists will always make the between batches distance larger
    knn_dists2 = np.zeros_like(knn_dists)
    for batch in range(len(X)):
        knn_dists2[:, sum(b_neighbors[0:batch]):sum(b_neighbors[0:(batch+1)])] = knn_dists[:, :b_neighbors[batch]] 

    # Modify pairwise distance matrix where the elements are changed due to knn_dists
 
    # OLD
    # pair_dist[np.arange(pair_dist.shape[0])[:, None], knn_indices] = knn_dists2
    # #Ensure pairwise distance matrix is symmetric
    # pair_dist[knn_indices.T, np.arange(pair_dist.shape[1])[None, :]] = knn_dists2.T
    # # maker sure that diagonal is 0
    # pair_dist = np.triu(pair_dist, 1)
    # pair_dist += pair_dist.T

    # NEW, UMAP
    pairwise_distances = np.zeros_like(pair_dist)
    pairwise_distances[np.arange(pairwise_distances.shape[0])[:, None], knn_indices] = knn_dists2
    pairwise_distances = pairwise_distances + pairwise_distances.T - pairwise_distances * pairwise_distances.T

    end_time = time.time()
    print("modify distance 2, time used {:.4f}s".format(end_time - start_time))

    # start_time = time.time()     
    pairwise_distances = sp.csr_matrix(pairwise_distances)
    # end_time = time.time()
    # print("make sparse, time used {:.4f}s".format(end_time - start_time))
    # pairwise_distances = pair_dist
    return pairwise_distances, knn_indices, knn_dists2


def post_nn_distance3(X, n_neighbors, njobs = None):
    # get a pairwise distance matrix for all batches
    from sklearn.metrics import pairwise_distances
    start_time = time.time()
    pair_dist = pairwise_distances(np.concatenate(X, axis = 0), n_jobs = njobs)
    end_time = time.time()
    print("calculating pairwise distance, time used {:.4f}s".format(end_time - start_time))

    # start_time = time.time()
    # pair_dist2 = squareform(pdist(np.concatenate(X, axis=0)))
    # end_time = time.time()
    # print("calculating pairwise distance, time used {:.4f}s".format(end_time - start_time))
    # assert np.allclose(pair_dist, pair_dist2)
    if n_neighbors < len(X):
        raise ValueError("the number of neighbors should not be smaller than the number of batches")

    # get the start points, end points and number of neighbors each batch
    start_time = time.time()
    start_point, end_point, b_ratios = [], [], []
    start = 0

    for batch in range(len(X)):
        start_point.append(start)
        b_ratios.append(len(X[batch])/len(pair_dist)) 
        start += len(X[batch])
        end_point.append(start-1)

    # compute the number of nearest neighbors for each sample in each batch of X
    # If the number of neighbors is too small, the it might be that some batches don't have neighbors, for both sampling and fixed
    # b_neighbors = np.random.multinomial(n_neighbors, b_ratios)
    # assign b_neighbors directly according to proportion.
    b_neighbors = []
    for batch in range(len(X)):
        if batch == len(X) - 1:
            b_neighbors.append(max(int(n_neighbors - sum(b_neighbors)), 1))
        else:
            b_neighbors.append(max(int(n_neighbors * b_ratios[batch]), 1))

    # compute knn_indices based on b_neighbors
    knn_indices = np.zeros((len(pair_dist), n_neighbors))
    knn_dists = np.zeros((len(pair_dist), n_neighbors))
    for batch in range(len(X)):
        knn_indices[:, sum(b_neighbors[0:batch]):sum(b_neighbors[0:(batch+1)])] = fast_knn_indices(pair_dist[:, start_point[batch]:end_point[batch]+1], b_neighbors[batch]) + start_point[batch]
    knn_indices = knn_indices.astype(int)
    knn_dists = pair_dist[np.arange(pair_dist.shape[0])[:, None], knn_indices].copy()
    end_time = time.time()
    print("knn separate, time used {:.4f}s".format(end_time - start_time))
    
    start_time = time.time()
    for batch_i in range(len(X)):
        ref_block = knn_dists[start_point[batch_i]:(end_point[batch_i] + 1), sum(b_neighbors[0:batch_i]):sum(b_neighbors[0:(batch_i+1)])]
        for batch_j in range(len(X)):
            if batch_i != batch_j:
                block = knn_dists[start_point[batch_i]:(end_point[batch_i] + 1), sum(b_neighbors[0:batch_j]):sum(b_neighbors[0:(batch_j+1)])]
                knn_dists[start_point[batch_i]:(end_point[batch_i] + 1), sum(b_neighbors[0:batch_j]):sum(b_neighbors[0:(batch_j+1)])] = block/np.mean(block, axis = 1, keepdims = True) * np.mean(ref_block, axis = 1, keepdims = True)
    end_time = time.time()
    print("modify distance 1, time used {:.4f}s".format(end_time - start_time))

    # quantile normalize
    start_time = time.time()
    knn_sort = np.argsort(knn_dists, axis = 1)
    knn_indices = knn_indices[np.arange(pair_dist.shape[0])[:, None], knn_sort]
    knn_dists = knn_dists[np.arange(pair_dist.shape[0])[:, None], knn_sort]

    ori_indices = fast_knn_indices(pair_dist, n_neighbors)
    knn_dists = pair_dist[np.arange(pair_dist.shape[0])[:, None], ori_indices].copy()    
    end_time = time.time()
    print("quantile normalize, time used {:.4f}s".format(end_time - start_time))

    start_time = time.time()
    # Modify pairwise distance matrix where the elements are changed due to knn_dists, 
    # pairwise_distance does not affect the UMAP visualization and leiden clustering

    # OLD
    # pair_dist[np.arange(pair_dist.shape[0])[:, None], knn_indices] = knn_dists2
    # #Ensure pairwise distance matrix is symmetric
    # pair_dist[knn_indices.T, np.arange(pair_dist.shape[1])[None, :]] = knn_dists2.T
    # # maker sure that diagonal is 0
    # pair_dist = np.triu(pair_dist, 1)
    # pair_dist += pair_dist.T

    # NEW, UMAP
    pairwise_distances = np.zeros_like(pair_dist)
    pairwise_distances[np.arange(pairwise_distances.shape[0])[:, None], knn_indices] = knn_dists
    pairwise_distances = pairwise_distances + pairwise_distances.T - pairwise_distances * pairwise_distances.T

    end_time = time.time()
    print("modify distance 2, time used {:.4f}s".format(end_time - start_time))

    # start_time = time.time()     
    pairwise_distances = sp.csr_matrix(pairwise_distances)
    # end_time = time.time()
    # print("make sparse, time used {:.4f}s".format(end_time - start_time))
    # pairwise_distances = pair_dist
    return pairwise_distances, knn_indices, knn_dists


def re_distance_nn(X, n_neighbors):
    """
    Calculate pairwise distance for X. Then, values of distances in blocks will be 
    replaced orderly by the reference block. The reference block is the largest size 
    of batch's block. 
    After that, we compute nearest neighbors for each sample in X. We will compute 
    the number of nearest neighbors picked from each batch first. Their sum must 
    equal to the input of n_neighbors This will be based on the size of batch. Then, 
    we compute knn_indices and knn_distance.

    Parameters
    -----------
    X: list, length is the number of batches
        The elements in X is arrays of shape of (n_samples, n_features)
        The input data to compute pairwise distance.

    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``. 


    Returns
    ------------
    pairwise_distances: csr_matrix (n_samples, n_samples)

    knn_indices

    knn_distance
    """


    # get a pairwise distance matrix
    pair_dist = squareform(pdist(np.concatenate(X, axis=0)))
    if n_neighbors < len(X):
        raise ValueError("the number of neighbors should not be smaller than the number of batches")

    # get the matrix who has the largest numbers of elements as the reference matrix
    start_point, end_point, b_ratios = [], [], []
    maxnum, maxbatch, start, number = 0, 0, 0, 0

    for batch in range(len(X)):
        start_point.append(start)
        b_ratios.append(len(X[batch])/len(pair_dist)) 
        number = len(X[batch])
        start += number
        end = start-1
        end_point.append(end)
        if number > maxnum:
            maxnum = start
            maxbatch = batch

    # pick the largest matrix as the reference matrix
    ref_dis = np.ravel(pair_dist[start_point[maxbatch]:end_point[maxbatch]+1,    
                                    start_point[maxbatch]:end_point[maxbatch]+1])

    # compute the number of nearest neighbors for each sample in each batch of X
    # b_neighbors = np.random.multinomial(n_neighbors, b_ratios)
    # assign b_neighbors directly according to proportion.
    b_neighbors = []
    for batch in range(len(X)):
        if batch == len(X) - 1:
            b_neighbors.append(max(int(n_neighbors - sum(b_neighbors)), 1))
        else:
            b_neighbors.append(max(int(n_neighbors * b_ratios[batch]), 1))


    # Modify distances for each block
    i = 0
    distance = np.zeros((len(pair_dist), len(pair_dist)), dtype=np.float32)
    for rows in range(len(X)):
        for batch in range(i, len(X)):
            if [rows, batch] != [maxbatch, maxbatch]:
                blocks = pair_dist[start_point[rows]:end_point[rows]+1, start_point[batch]:end_point[batch]+1].flatten()
                d = np.argsort(blocks)
                blocks[d] = list(range(len(d)))
    
                sample = np.random.choice(ref_dis, len(d), replace=False)
                sample.sort()
    
                blocks = sample[blocks.astype(int)]
                distance[start_point[rows]:end_point[rows]+1, start_point[batch]:end_point[batch]+1] = \
                    np.reshape(blocks, (end_point[rows]+1-start_point[rows], end_point[batch]+1-start_point[batch]))
            else:
                distance[start_point[rows]:end_point[rows]+1, start_point[batch]:end_point[batch]+1] = \
                    pair_dist[start_point[rows]:end_point[rows]+1, start_point[batch]:end_point[batch]+1]
        i += 1
        
    distance = np.triu(distance, 1)
    distance += distance.T
    pairwise_distances = sp.csr_matrix(distance)

    
    # compute knn_indices and knn_dists based on modified pairwise distances and 
    # customized number of nearest neighbors
    knn_indices = np.zeros((len(pair_dist), n_neighbors))
    for batch in range(len(X)):
        knn_indices[:, sum(b_neighbors[0:batch]):sum(b_neighbors[0:batch+1])] = \
                    fast_knn_indices(pair_dist[:, start_point[batch]:end_point[batch]+1], b_neighbors[batch]) \
                    + start_point[batch]
    
    knn_indices = knn_indices.astype(int)
    knn_dists = pair_dist[np.arange(pair_dist.shape[0])[:, None], knn_indices].copy()
    
    return pairwise_distances, knn_indices, knn_dists



def re_nn_distance(X, n_neighbors, njobs = None):
    """
    we compute nearest neighbors for each sample in X. We will compute 
    the number of nearest neighbors picked from each batch first. Their sum must 
    equal to the input of n_neighbors. This will be based on the size of batch. 
    Then, we compute knn_indices.
    knn_distance is computed based on original knn_indices, which is based on 
    n_neighbor rather than b_neighbors.

    Parameters
    -----------
    X: list, length is the number of batches
        The elements in X is arrays of shape of (n_samples, n_features)
        The input data to compute pairwise distance.

    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``. 


    Returns
    ------------
    pairwise_distances: csr_matrix (n_samples, n_samples)

    knn_indices

    knn_dists
    """

    # get a pairwise distance matrix for all batches
    # pair_dist = squareform(pdist(np.concatenate(X, axis=0)))
    from sklearn.metrics import pairwise_distances
    start_time = time.time()
    pair_dist = pairwise_distances(np.concatenate(X, axis = 0), n_jobs = njobs)
    end_time = time.time()
    print("Calculating pairwise distance, time used: {:.4f}s".format(end_time - start_time))

    if n_neighbors < len(X):
        raise ValueError("the number of neighbors should not be smaller than the number of batches")

    start_time = time.time()
    # get the start points, end points and size for each batch
    start_point, end_point, b_ratios = [], [], []
    start = 0

    for batch in range(len(X)):
        start_point.append(start)
        b_ratios.append(len(X[batch])/len(pair_dist)) 
        start += len(X[batch])
        end_point.append(start-1)

    # compute the number of nearest neighbors for each sample in each batch of X
    # If the number of neighbors is too small, the it might be that some batches don't have neighbors, for both sampling and fixed
    # b_neighbors = np.random.multinomial(n_neighbors, b_ratios)
    # assign b_neighbors directly according to proportion.
    b_neighbors = []
    for batch in range(len(X)):
        if batch == len(X) - 1:
            b_neighbors.append(max(int(n_neighbors - sum(b_neighbors)), 1))
        else:
            b_neighbors.append(max(int(n_neighbors * b_ratios[batch]), 1))

    # compute knn_indices based on b_neighbors
    knn_indices = np.zeros((len(pair_dist), n_neighbors))
    for batch in range(len(X)):
        knn_indices[:, sum(b_neighbors[0:batch]):sum(b_neighbors[0:batch+1])] = \
            fast_knn_indices(pair_dist[:, start_point[batch]:end_point[batch]+1], b_neighbors[batch]) + start_point[batch]
    
    knn_indices = knn_indices.astype(int)
    end_time = time.time()
    print("Find k-nearest neighbor within each batches, time used: {:.4f}s".format(end_time - start_time))

    # Only modify distance for points in knn_indices
    # Get knn_distance from original knn_indices, which is nearest neighbors totally 
    # based on n_neighbors 
    start_time = time.time()
    ori_indices = fast_knn_indices(pair_dist, n_neighbors)
    knn_dists = pair_dist[np.arange(pair_dist.shape[0])[:, None], ori_indices].copy()
    # Modify pairwise distance matrix where the elements are changed due to knn_dists
    pair_dist[np.arange(pair_dist.shape[0])[:, None], knn_indices] = knn_dists

    #Ensure pairwise distance matrix is symmetric
    pair_dist[knn_indices.T, np.arange(pair_dist.shape[1])[None, :]] = knn_dists.T
    pair_dist = np.triu(pair_dist, 1)
    pair_dist += pair_dist.T
    end_time = time.time()
    print("Find k-nearest neighbor across each batches, time used: {:.4f}s".format(end_time - start_time))

    pairwise_distances = sp.csr_matrix(pair_dist)

    return pairwise_distances, knn_indices, knn_dists



'''