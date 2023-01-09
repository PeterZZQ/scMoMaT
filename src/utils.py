import numpy as np
import matplotlib.pyplot as plt


import numpy as np
from scipy import stats
import scipy.sparse as sp
from umap.utils import fast_knn_indices
import time
try:
    from adjustText import adjust_text
except ImportError:
    adjust_text_flag = False

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
            ax.scatter(zs[batch][:,0], zs[batch][:,1], color = colormap(batch), label = "batch " + str(batch + 1), s = _kwargs["s"], alpha = _kwargs["alpha"])
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
            if adjust_text_flag:
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
                    # if exist cells
                    if zs[batch][index,0].shape[0] > 0:
                        texts.append(axs[batch].text(np.median(zs[batch][index,0]), np.median(zs[batch][index,1]), color = "black", s = cluster_types[i], fontsize = _kwargs["text_size"], weight = 'semibold', in_layout = True))
            
            axs[batch].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = (len(cluster_types) // 15) + 1, bbox_to_anchor=(0.94, 1), markerscale = _kwargs["markerscale"])
            axs[batch].set_title("batch " + str(batch + 1), fontsize = 25)

            axs[batch].tick_params(axis = "both", which = "major", labelsize = 15)

            axs[batch].set_xlabel(axis_label + " 1", fontsize = 19)
            axs[batch].set_ylabel(axis_label + " 2", fontsize = 19)

            axs[batch].spines['right'].set_visible(False)
            axs[batch].spines['top'].set_visible(False)  

            axs[batch].set_xlim(np.min(np.concatenate([x[:,0] for x in zs])), np.max(np.concatenate([x[:,0] for x in zs])))
            axs[batch].set_ylim(np.min(np.concatenate([x[:,1] for x in zs])), np.max(np.concatenate([x[:,1] for x in zs])))

            if label_inplace:
                if adjust_text_flag:
                    adjust_text(texts, only_move={'points':'xy', 'texts':'xy'})        
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
            axs[clust%nrows, clust//nrows].set_ylabel("feature score", fontsize = 19)
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
            axs.set_ylabel("feature score", fontsize = 19)
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
            axs[clust].set_ylabel("feature score", fontsize = 19)
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
            _ = axs[marker%nrows, marker//nrows].set_ylabel("feature score", fontsize = 20)
            axs[marker%nrows, marker//nrows].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
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
            _ = axs.set_ylabel("feature score", fontsize = 20)
            axs.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

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
            _ = axs[marker].set_ylabel("feature score", fontsize = 20)
            axs[marker].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

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

def post_process(X, n_neighbors, r = None, njobs = None, return_sparse_dist = False):
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
        # fast_knn_indices include the node itself, it sort the values too
        knn_indices[:, sum(b_neighbors[0:batch]):sum(b_neighbors[0:(batch+1)])] = fast_knn_indices(pair_dist[:, start_point[batch]:end_point[batch]+1], b_neighbors[batch] + 1)[:, 1:] + start_point[batch]
    knn_indices = knn_indices.astype(int)
    knn_dists = pair_dist[np.arange(pair_dist.shape[0])[:, None], knn_indices].copy()

    if r is not None:
        # construct within batch knn graph
        knn_indices2 = np.zeros((len(pair_dist), n_neighbors))
        for batch in range(len(X)):
            # fast_knn_indices include the node itself, it sort the values too
            knn_indices2[start_point[batch]:end_point[batch]+1, :] = fast_knn_indices(pair_dist[start_point[batch]:end_point[batch]+1, start_point[batch]:end_point[batch]+1], n_neighbors + 1)[:, 1:] + start_point[batch]
        knn_indices2 = knn_indices2.astype(int)
        knn_dists2 = pair_dist[np.arange(pair_dist.shape[0])[:, None], knn_indices2].copy()
        # mask matrix stores the neighbors that need to be removed
        mask = np.zeros((len(pair_dist), n_neighbors))
        # radius cut-off, useful for disproportionate cell type composition
        for batch_i in range(len(X)):
            for batch_j in range(len(X)):
                if batch_i != batch_j:
                    # r = segment1d(knn_dists[start_point[batch_i]:(end_point[batch_i] + 1), sum(b_neighbors[0:batch_j]):sum(b_neighbors[0:(batch_j+1)])].reshape(-1))
                    # radius = r * np.median(knn_dists[start_point[batch_i]:(end_point[batch_i] + 1), sum(b_neighbors[0:batch_j]):sum(b_neighbors[0:(batch_j+1)])].reshape(-1))
                    # print("batch i: {:d}, batch j: {:d}, r: {:.4f}".format(batch_i, batch_j, radius))
                    assert r <= 1
                    radius = np.percentile(knn_dists[start_point[batch_i]:(end_point[batch_i] + 1), sum(b_neighbors[0:batch_j]):sum(b_neighbors[0:(batch_j+1)])].reshape(-1), r * 100)
                    print("batch i: {:d}, batch j: {:d}, r: {:.4f}".format(batch_i, batch_j, radius))
                    
                    # the removed neighbors are denoted as 1
                    mask_block = (knn_dists[start_point[batch_i]:(end_point[batch_i] + 1), sum(b_neighbors[0:batch_j]):sum(b_neighbors[0:(batch_j+1)])] > radius)
                    mask[start_point[batch_i]:(end_point[batch_i] + 1), sum(b_neighbors[0:batch_j]):sum(b_neighbors[0:(batch_j+1)])] = mask_block
                    removed_neighs = np.where(np.sum(mask_block, axis = 1) > 0)[0] + start_point[batch_i]

                    # update knn_indices
                    knn_indices[np.ix_(removed_neighs, np.arange(sum(b_neighbors[0:batch_j]), sum(b_neighbors[0:(batch_j+1)])))] = \
                        knn_indices2[np.ix_(removed_neighs, np.arange(sum(b_neighbors[0:batch_j]), sum(b_neighbors[0:(batch_j+1)])))]
                    
                    # update knn_dists
                    knn_dists[np.ix_(removed_neighs, np.arange(sum(b_neighbors[0:batch_j]), sum(b_neighbors[0:(batch_j+1)])))] = \
                        knn_dists2[np.ix_(removed_neighs, np.arange(sum(b_neighbors[0:batch_j]), sum(b_neighbors[0:(batch_j+1)])))]
        

    end_time = time.time()
    print("knn separate, time used {:.4f}s".format(end_time - start_time))
    
    start_time = time.time()
    for batch_i in range(len(X)):
        ref_block = knn_dists[start_point[batch_i]:(end_point[batch_i] + 1), sum(b_neighbors[0:batch_i]):sum(b_neighbors[0:(batch_i+1)])]
        for batch_j in range(len(X)):
            if batch_i != batch_j:
                block = knn_dists[start_point[batch_i]:(end_point[batch_i] + 1), sum(b_neighbors[0:batch_j]):sum(b_neighbors[0:(batch_j+1)])]
                knn_dists[start_point[batch_i]:(end_point[batch_i] + 1), sum(b_neighbors[0:batch_j]):sum(b_neighbors[0:(batch_j+1)])] = block/(np.mean(block, axis = 1, keepdims = True) + 1e-6) * np.mean(ref_block, axis = 1, keepdims = True)
    end_time = time.time()
    print("modify distance 1, time used {:.4f}s".format(end_time - start_time))

    start_time = time.time()
    # Modify pairwise distance matrix where the elements are changed due to knn_dists, 
    # pairwise_distance does not affect the UMAP visualization and leiden clustering


    # NEW, UMAP
    if return_sparse_dist:
        pairwise_distances = np.zeros_like(pair_dist)
        pairwise_distances[np.arange(pairwise_distances.shape[0])[:, None], knn_indices] = knn_dists
        pairwise_distances = pairwise_distances + pairwise_distances.T - pairwise_distances * pairwise_distances.T

        end_time = time.time()
        print("modify distance 2, time used {:.4f}s".format(end_time - start_time))

        # start_time = time.time()     
        pair_dist = sp.csr_matrix(pairwise_distances)
    else:
        pair_dist = sp.csr_matrix(pair_dist)
    # end_time = time.time()
    # print("make sparse, time used {:.4f}s".format(end_time - start_time))
    # pairwise_distances = pair_dist
    return pair_dist, knn_indices, knn_dists
