# In[]
import sys, os
sys.path.append('../')
sys.path.append('../src/')

import torch
import numpy as np
import utils
from torch.nn import Module, Parameter
import torch.optim as opt
from utils import preprocess
import torch.nn.functional as F

import torch.optim as opt
from torch import softmax, log_softmax, Tensor
from sklearn.cluster import KMeans


from sklearn.decomposition import PCA
from umap_batch import UMAP
from utils import re_nn_distance, re_distance_nn

import pandas as pd  
import scipy.sparse as sp
import model
import time

import quantile 

import coupleNMF as coupleNMF

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

# In[]
import importlib 
importlib.reload(model)

# In[]
# read in dataset
dir = '../data/real/diag/BMMC/small_ver/'
result_dir = "bmmc/"

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
        
        # PLOT FUNCTION
        x_pca = PCA(n_components = 100).fit_transform(counts_atac)
        x_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(x_pca)
        utils.plot_latent_ext([x_umap], annos = [labels[-1]], mode = "joint", save = result_dir + f'X_batch{batch}_atac.png', figsize = (10,7), axis_label = "UMAP")
        
    except:
        counts_atac = None
        
    try:
        counts_rna = np.array(sp.load_npz(os.path.join(dir, 'GxC' + str(batch) + ".npz")).todense().T)
        print("RNA")
        print(counts_rna.shape)
        print(counts_rna)
        counts_rna = utils.preprocess(counts_rna, modality = "RNA")

        # PLOT FUNCTION
        x_pca = PCA(n_components = 100).fit_transform(counts_rna)
        x_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(x_pca)
        utils.plot_latent_ext([x_umap], annos = [labels[-1]], mode = "joint", save = result_dir + f'X_batch{batch}_rna.png', figsize = (10,7), axis_label = "UMAP")

    except:
        counts_rna = None
    
    # preprocess the count matrix
    counts_rnas.append(counts_rna)
    counts_atacs.append(counts_atac)

counts = {"rna":counts_rnas, "atac": counts_atacs}

A = sp.load_npz(os.path.join(dir, 'GxR.npz'))
A = np.array(A.todense())
# normalize A
A = utils.preprocess(A, modality = "interaction")
interacts = {"rna_atac": A}


# obtain the feature name
genes = pd.read_csv(dir + "genes.txt", header = None).values.squeeze()
regions = pd.read_csv(dir + "regions.txt", header = None).values.squeeze()

feats_name = {"rna": genes, "atac": regions}
counts["feats_name"] = feats_name

# new
counts["rna"][1] = counts["atac"][1] @ interacts["rna_atac"].T
# normalization? 
# quantile
# counts["rna"][1] = quantile_norm(targ_mtx = counts["rna"][1], ref_mtx = counts["rna"][0], replace = True)
counts["rna"][1] = utils.preprocess(counts["rna"][1], modality = "RNA")
# normalization method below is not good
# counts["rna"][1] = counts["rna"][1]/(np.sum(counts["rna"][1], axis = 1, keepdims = True) + 1e-12)
# counts["rna"][1] = np.log1p(counts["rna"][1])
# counts["rna"][1] = counts["rna"][1]/np.max(counts["rna"][1])
interacts = None

# In[]
#hyper parameters: best lr = 5e-3, T = 4000, latent_dims = 13
alpha = [1000, 1, 5]
# alpha = [1000, 0, 0]
batchsize = 0.1
run = 0
K = 30
Ns = [K] * 2
N_feat = Ns[0]
interval = 1000
T = 6000
lr = 1e-2

# use interaction matrix
# start_time = time.time()
model1 = model.cfrm_vanilla(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
# losses1 = model1.train_func(T = T)
# x = np.linspace(0, T, int(T/interval))
# plt.plot(x, losses1)
# end_time = time.time()
# print("running time: " + str(end_time - start_time))

# torch.save(model1.state_dict(), result_dir + f'real_{K}_{T}_nointeract.pt')
model1.load_state_dict(torch.load(result_dir + f'real_{K}_{T}_nointeract.pt'))

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

utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = None, figsize = (10,15), axis_label = "UMAP")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = None, figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = None, figsize = (15,10), axis_label = "UMAP", markerscale = 6)

# In[]
n_neighbors = 10

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

s_pair_dist, knn_indices, knn_dists = re_nn_distance(zs, n_neighbors)

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
        
utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = result_dir + f'latent_separate.png', 
                      figsize = (10,15), axis_label = "Latent")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = result_dir + f'latent_batches.png', 
                      figsize = (10,10), axis_label = "Latent", markerscale = 6)

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = result_dir + f'latent_clusters.png', 
                      figsize = (10,10), axis_label = "Latent", markerscale = 6)


# In[] Baseline methods
# 1. Seurat
seurat_path = "bmmc_healthyhema_1000"
# %%
