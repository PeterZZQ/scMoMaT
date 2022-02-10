# In[]
import sys, os
sys.path.append('../')
sys.path.append('../src/')

import torch
import numpy as np
import utils

from sklearn.decomposition import PCA
from umap_batch import UMAP

import pandas as pd  
import scipy.sparse as sp
import model
import time

import quantile 

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def quantile_norm(targ_mtx, ref_mtx, replace = False):
    # sampling and don't put back
    reference = np.sort(np.random.choice(ref_mtx.reshape(-1), targ_mtx.shape[0] * targ_mtx.shape[1], replace = replace))
    dist_temp = targ_mtx.reshape(-1)
    dist_idx = np.argsort(dist_temp)
    dist_temp[dist_idx] = reference
    return dist_temp.reshape(targ_mtx.shape[0], targ_mtx.shape[1])

import importlib 
importlib.reload(model)

# In[]
# read in dataset
dir = '../data/real/diag/Xichen/'
result_dir = "spleen/cfrm_liger_quantile/"
seurat_path = "spleen/seurat/"
liger_path = "spleen/liger/"

counts_rnas = []
counts_atacs = []
counts_proteins = []
labels = []
for batch in [1, 2]:
    labels.append(pd.read_csv(os.path.join(dir, 'meta_c' + str(batch) + '.csv'), index_col=0)["cell_type"].values.squeeze())
    
    try:
        counts_atac = np.array(sp.load_npz(os.path.join(dir, 'RxC' + str(batch) + ".npz")).todense().T)
        print("ATAC")
        print(counts_atac.shape)
        counts_atac = utils.preprocess(counts_atac, modality = "ATAC")
        print("sparsity: {:.4f}".format(np.sum(counts_atac != 0)/(counts_atac.shape[0] * counts_atac.shape[1])))
        
        # # PLOT FUNCTION
        # x_pca = PCA(n_components = 100).fit_transform(counts_atac)
        # x_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(x_pca)
        # utils.plot_latent_ext([x_umap], annos = [labels[-1]], mode = "joint", save = result_dir + f'X_batch{batch}_atac.png', figsize = (10,7), axis_label = "UMAP")
        
    except:
        counts_atac = None
        
    try:
        counts_rna = np.array(sp.load_npz(os.path.join(dir, 'GxC' + str(batch) + ".npz")).todense().T)
        print("RNA")
        print(counts_rna.shape)
        counts_rna = utils.preprocess(counts_rna, modality = "RNA", log = False)
        # counts_rna = utils.preprocess_liger(counts_rna, with_mean = False)

        # # PLOT FUNCTION
        # x_pca = PCA(n_components = 100).fit_transform(counts_rna)
        # x_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(x_pca)
        # utils.plot_latent_ext([x_umap], annos = [labels[-1]], mode = "joint", save = result_dir + f'X_batch{batch}_rna.png', figsize = (10,7), axis_label = "UMAP")

    except:
        counts_rna = None
    
    # preprocess the count matrix
    counts_rnas.append(counts_rna)
    counts_atacs.append(counts_atac)

counts = {"rna":counts_rnas, "atac": counts_atacs}

A = sp.load_npz(os.path.join(dir, 'GxR.npz'))
A = np.array(A.todense())
# normalize A
# A = utils.preprocess(A, modality = "interaction")
interacts = {"rna_atac": A}


# obtain the feature name
genes = pd.read_csv(dir + "genes.txt", header = None).values.squeeze()
regions = pd.read_csv(dir + "regions.txt", header = None).values.squeeze()

feats_name = {"rna": genes, "atac": regions}
counts["feats_name"] = feats_name


# CALCULATE PSEUDO-SCRNA-SEQ
counts["rna"][1] = counts["atac"][1] @ interacts["rna_atac"].T
# NORMALIZATION OF PSEUDO-SCRNA-SEQ
counts["rna"][1] = utils.preprocess(counts["rna"][1], modality = "RNA", log = False)
# LIGER flavor
# counts["rna"][1] = utils.preprocess_liger(counts["rna"][1], with_mean = False)
# quantile
# counts["rna"][1] = quantile_norm(targ_mtx = counts["rna"][1], ref_mtx = counts["rna"][0], replace = True)

# # PLOT FUNCTION
# x_pca = PCA(n_components = 100).fit_transform(counts["rna"][1])
# x_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(x_pca)
# utils.plot_latent_ext([x_umap], annos = [labels[-1]], mode = "joint", save = None, figsize = (10,7), axis_label = "UMAP")

# HISTOGRAM
for count in counts["rna"]:
    fig = plt.figure()
    ax = fig.add_subplot()
    _ = ax.hist(count.reshape(-1), bins = 50)

interacts = None
# counts = {"rna": counts["rna"], "feats_name": feats_name}
# In[]
#hyper parameters: best lr = 5e-3, T = 4000, latent_dims = 13
alpha = [1000, 1, 5]
# alpha = [1000, 0, 0]
batchsize = 0.1
run = 0
K = 15
Ns = [K] * 2
N_feat = Ns[0]
interval = 1000
T = 6000
lr = 1e-2

# use interaction matrix
start_time = time.time()
model1 = model.cfrm_vanilla(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
losses1 = model1.train_func(T = T)
end_time = time.time()
print("running time: " + str(end_time - start_time))

x = np.linspace(0, T, int(T/interval) + 1)
plt.plot(x, losses1)

torch.save(model1.state_dict(), result_dir + f'CFRM_{K}_{T}.pt')
model1.load_state_dict(torch.load(result_dir + f'CFRM_{K}_{T}.pt'))

# In[]
umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4, random_state = 0) 
zs = []
labels = []
prec_labels = []
pre_labels = []
for batch in range(0,2):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    # z = model1.C_cells[batch].cpu().detach().numpy()
    zs.append(z)
    labels.append(pd.read_csv(os.path.join(dir, 'meta_c' + str(batch + 1) + '.csv'), index_col=0)["cell_type"].values.squeeze())
    pre_labels.append(np.argmax(z, axis = 1).squeeze())

x_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))
# separate into batches
x_umaps = []
for batch in range(0,2):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
    elif batch == 1:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = result_dir + f'latent_separate_{K}_{T}.png', figsize = (10,15), axis_label = "UMAP")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = result_dir + f'latent_batches_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = result_dir + f'latent_clusters_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6)


# In[]
from scipy.stats import spearmanr, pearsonr
ave_spearman = 0
ave_pearson = 0
counts = 0
for cluster in np.sort(np.unique(labels[0])):
    idx_0 = np.where(labels[0] == cluster)[0]
    idx_1 = np.where(labels[1] == cluster)[0]
    if len(idx_0) == 0 or len(idx_1) == 0:
        continue
    clust_center0 = np.mean(zs[0][idx_0, :], axis = 0)
    clust_center1 = np.mean(zs[1][idx_1, :], axis = 0)
    spearman, pval = spearmanr(clust_center0, clust_center1)
    pearson, pval = pearsonr(clust_center0, clust_center1)
    print(cluster)
    print('Spearman: {:.2f}'.format(spearman))
    print('Pearson: {:.2f}'.format(pearson))
    ave_pearson += pearson
    ave_spearman += spearman
    counts += 1

ave_pearson /= counts
ave_spearman /= counts
print('Average spearman: {:.2f}'.format(spearman))
print('Average pearson: {:.2f}'.format(pearson))

label0 = labels[0]
label1 = labels[1]
zs0 = zs[0]
zs1 = zs[1]
b_foll0 = np.where(label0 == "B_follicular")[0]
b_foll1 = np.where(label1 == "B_follicular")[0]
b_foll0_center = np.mean(zs0[b_foll0, :], axis = 0)
b_foll1_center = np.mean(zs1[b_foll1, :], axis = 0)
plt.bar(x = np.arange(K), height = b_foll0_center)
plt.show()
plt.bar(x = np.arange(K), height = b_foll1_center)
plt.show()

H1 = pd.read_csv(liger_path + "liger_c1.csv", sep = ",", index_col = 0).values
H2 = pd.read_csv(liger_path + "liger_c2.csv", sep = ",", index_col = 0).values
label0 = labels[0]
label1 = labels[1]
zs0 = zs[0]
zs1 = zs[1]
b_foll0 = np.where(label0 == "B_follicular")[0]
b_foll1 = np.where(label1 == "B_follicular")[0]
b_foll0_center = np.mean(H1[b_foll0, :], axis = 0)
b_foll1_center = np.mean(H2[b_foll1, :], axis = 0)
plt.bar(x = np.arange(b_foll0_center.shape[0]), height = b_foll0_center)
plt.show()
plt.bar(x = np.arange(b_foll1_center.shape[0]), height = b_foll1_center)
plt.show()

# In[]
n_neighbors = 30

zs = []
for batch in range(0,2):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    # z = model1.C_cells[batch].cpu().detach().numpy()
    zs.append(z)

s_pair_dist, knn_indices, knn_dists = utils.re_nn_distance(zs, n_neighbors)
# s_pair_dist, knn_indices, knn_dists = utils.re_distance_nn(zs, n_neighbors)

umap_op = UMAP(n_components = 2, n_neighbors = n_neighbors, min_dist = 0.4, random_state = 0, 
                metric='precomputed', knn_dists=knn_dists, knn_indices=knn_indices)
x_umap = umap_op.fit_transform(s_pair_dist)
# separate into batches
x_umaps = []
for batch in range(0,2):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
    elif batch == 1:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = result_dir + f'latent_separate_{K}_processed.png', 
                      figsize = (10,15), axis_label = "Latent")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = result_dir + f'latent_batches_{K}_processed.png', 
                      figsize = (15,10), axis_label = "Latent", markerscale = 6)

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = result_dir + f'latent_clusters_{K}_processed.png', 
                      figsize = (15,10), axis_label = "Latent", markerscale = 6)


# In[] Baseline methods
"""
# 1. Seurat
seurat_pcas = [pd.read_csv(seurat_path + "seurat_pca_c1.txt", sep = "\t", index_col = 0).values, 
               pd.read_csv(seurat_path + "seurat_pca_c2.txt", sep = "\t", index_col = 0).values]
seurat_umaps = [pd.read_csv(seurat_path + "seurat_umap_c1.txt", sep = "\t", index_col = 0).values,
               pd.read_csv(seurat_path + "seurat_umap_c2.txt", sep = "\t", index_col = 0).values]


utils.plot_latent_ext(seurat_umaps, annos = labels, mode = "separate", save = seurat_path + f'latent_separate_seurat.png', 
                      figsize = (10,15), axis_label = "Latent")

utils.plot_latent_ext(seurat_umaps, annos = labels, mode = "modality", save = seurat_path + f'latent_batches_seurat.png', 
                      figsize = (10,10), axis_label = "Latent", markerscale = 6)

utils.plot_latent_ext(seurat_umaps, annos = labels, mode = "joint", save = seurat_path + f'latent_clusters_seurat.png', 
                      figsize = (10,10), axis_label = "Latent", markerscale = 6)

# 2. Liger
H1 = pd.read_csv(liger_path + "liger_c1_norm.csv", sep = ",", index_col = 0).values
H2 = pd.read_csv(liger_path + "liger_c2_norm.csv", sep = ",", index_col = 0).values
liger_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(np.concatenate((H1, H2), axis = 0))
liger_umaps = []
for batch in range(0,2):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + counts["rna"][batch].shape[0]
        liger_umaps.append(liger_umap[start_pointer:end_pointer,:])
    elif batch == 1:
        start_pointer = start_pointer + counts["rna"][batch - 1].shape[0]
        liger_umaps.append(liger_umap[start_pointer:,:])
    else:
        start_pointer = start_pointer + counts["rna"][batch - 1].shape[0]
        end_pointer = start_pointer + counts["rna"][batch].shape[0]
        liger_umaps.append(liger_umap[start_pointer:end_pointer,:])

utils.plot_latent_ext(liger_umaps, annos = labels, mode = "separate", save = liger_path + f'latent_separate_liger.png', 
                      figsize = (10,15), axis_label = "Latent")

utils.plot_latent_ext(liger_umaps, annos = labels, mode = "modality", save = liger_path + f'latent_batches_liger.png', 
                      figsize = (10,10), axis_label = "Latent", markerscale = 6)

utils.plot_latent_ext(liger_umaps, annos = labels, mode = "joint", save = liger_path + f'latent_clusters_liger.png', 
                      figsize = (10,10), axis_label = "Latent", markerscale = 6)
"""
# %%
