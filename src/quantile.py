# In[0]
import numpy as np 
from scipy.interpolate import interp1d 
from sklearn.neighbors import NearestNeighbors

# In[1]
def quantile_norm(Cs, min_cells = 20, max_sample = 1000, quantiles = 50, ref = None, refine = False):
    n_batches = len(Cs)
    n_clusts = Cs[0].shape[1]

    # find reference batch, maximum number of cells
    if ref is None:
        num_cells = [C.shape[0] for C in Cs]
        ref = np.argmax(num_cells)

    clusters = [np.argmax(C, axis = 1).squeeze() for C in Cs]

    # TODO: liger SNF refine cluster assignment, construct k nearest neighbor and check the most happening neighborhood cluster assignment, as current cluster assignment
    if refine:
        for batch in range(n_batches):
            nbrs = NearestNeighbors(n_neighbors=5).fit(Cs[batch])
            _, knn = nbrs.kneighbors(Cs[batch])
            cluster_votes = clusters[batch][knn.reshape(-1)].reshape(Cs[batch].shape[0], -1)
            clusters[batch] = []
            for n in range(cluster_votes.shape[0]):
                cluster_unique, counts = np.unique(cluster_votes[n,:], return_counts = True)
                clusters[batch].append(cluster_unique[np.argmax(counts)])
            clusters[batch] = np.array(clusters[batch])

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

                # add values in order to make interp1d works
                q1 = np.concatenate(([np.min(Cs[batch][cell_idx, clust])], q1, [np.max(Cs[batch][cell_idx, clust])]))
                q2 = np.concatenate(([np.min(Cs[ref][cell_ref_idx, clust])], q2, [np.max(Cs[ref][cell_ref_idx, clust])]))

                if np.sum(q1) == 0 or np.sum(q2) == 0 or len(np.unique(q1)) < 2 or len(np.unique(q2)) < 2:
                    Cs[batch][cell_idx, clust] = 0 
                else:
                    f = interp1d(q2, q1)
                    Cs[batch][cell_idx, clust] = f(Cs[batch][cell_idx, clust])
    
    return Cs


# In[2]
# import pandas as pd
# import os 
# import torch
# from umap import UMAP
# import utils

# C1 = pd.read_csv("./C1.txt", index_col = 0, header = None).values
# C2 = pd.read_csv("./C2.txt", index_col = 0, header = None).values

# Cs = [C1, C2]

# Cs = quantile_norm(Cs, min_cells = 0, refine = False)

# dir = '../data/simulated/'

# path = '2b4c_sigma0.1_b2_1/'
# # path = '2b5c_ziqi1/'

# counts_rna1 = pd.read_csv(os.path.join(dir + path, 'GxC1.txt'), sep = "\t", header = None).values.T
# counts_rna2 = pd.read_csv(os.path.join(dir + path, 'GxC2.txt'), sep = "\t", header = None).values.T
# counts_atac1 = pd.read_csv(os.path.join(dir + path, 'RxC1.txt'), sep = "\t", header = None).values.T
# counts_atac2 = pd.read_csv(os.path.join(dir + path, 'RxC2.txt'), sep = "\t", header = None).values.T
# A = pd.read_csv(os.path.join(dir + path, 'region2gene.txt'), sep = "\t", header = None).values.T

# counts_rna1 = np.array(counts_rna1)
# counts_rna2 = np.array(counts_rna2)
# counts_atac1 = np.array(counts_atac1)
# counts_atac2 = np.array(counts_atac2)
# A = np.array(A)

# label_rna = pd.read_csv(os.path.join(dir + path, "cell_label1.txt"), index_col=0, sep = "\t")["pop"].values.squeeze()
# label_atac = pd.read_csv(os.path.join(dir + path, "cell_label2.txt"), index_col=0, sep = "\t")["pop"].values.squeeze()

# counts = {"rna":[counts_rna1, None], "atac": [None, counts_atac2], "gact": A}


# Cs[0] = torch.softmax(torch.FloatTensor(Cs[0]), dim = 1).cpu().detach().numpy()
# Cs[1] = torch.softmax(torch.FloatTensor(Cs[1]), dim = 1).cpu().detach().numpy()

# max_rna = np.argmax(Cs[0], axis = 1).squeeze()
# max_atac = np.argmax(Cs[1], axis = 1).squeeze()

# umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4) 
# z = umap_op.fit_transform(np.concatenate((Cs[0], Cs[1]), axis = 0))

# utils.plot_latent(z[:Cs[0].shape[0],:], z[Cs[0].shape[0]:,:], max_rna, max_atac, mode= "separate", save = None)


# %%
