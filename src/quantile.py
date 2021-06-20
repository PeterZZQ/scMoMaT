import numpy as np 
from scipy.interpolate import interp1d 
from sklearn.neighbors import NearestNeighbors

def quantile_norm(Cs, min_cells = 20, max_sample = 1000, quantiles = 50, ref = None, refine = False):
    n_batches = len(Cs)
    n_clusts = Cs[0].shape[1]

    # find reference batch, maximum number of cells
    if ref is None:
        num_cells = [C.shape[0] for C in Cs]
        ref = np.argmax(num_cells)
        max_count = num_cells[ref]

    clusters = [np.argmax(C, axis = 1).squeeze() for C in Cs]

    # TODO: liger SNF refine cluster assignment, construct k nearest neighbor and check the most happening neighborhood cluster assignment, as current cluster assignment
    if refine:
        for batch in range(n_batches):
            nbrs = NearestNeighbors(n_neighbors=5).fit(Cs[batch])
            _, knn = nbrs.kneighbors(Cs[batch])
            cluster_votes = clusters[batch][knn.reshape(-1)].reshape(Cs[batch].shape[0], -1)
            clusters_assign, cluster_votes = np.unique(cluster_votes, axis = 1, return_counts = True)
            clusters = clusters_assign[np.ix_(np.arange(Cs[batch].shape[0]), np.argmax(cluster_votes, axis = 1))].squeeze()

    # Quantile_normalization
    for clust in range(n_clusts):
        cell_ref_idx, *_ = np.where(clusters[ref] == clust)
        for batch in range(n_batches):
            cell_idx, *_ = np.where(clusters[batch] == clust)
            if (len(cell_ref_idx) < min_cells) | (len(cell_idx) < min_cells):
                pass
            elif len(cell_idx) == 1:
                Cs[batch][cell_idx, clust] = np.mean(Cs[ref][cell_ref_idx, clust]) 
            else:
                # current batch
                q2 = np.quantile(np.random.choice(Cs[batch][cell_idx, clust], min(len(cell_idx), max_sample)), np.arange(0,1,1/quantiles))
                # reference batch
                q1 = np.quantile(np.random.choice(Cs[ref][cell_ref_idx, clust], min(len(cell_ref_idx), max_sample)), np.arange(0,1,1/quantiles))
                if np.sum(q1) == 0 | np.sum(q2) == 0 | len(np.unique(q1)) < 2 | len(np.unique(q2)) < 2:
                    Cs[batch][cell_idx, clust] = 0 
                else:
                    f = interp1d(q2, q1)
                    Cs[batch][cell_idx, clust] = f(Cs[batch][cell_idx, clust])
    
    return Cs

    