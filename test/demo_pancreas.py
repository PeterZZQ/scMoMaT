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
dir = '../data/real/hori/Pancreas/'
result_dir = "pancreas/cfrm_quantile/"

counts_rnas = []
counts_atacs = []
counts_proteins = []
labels = []
for batch in range(8):
    labels.append(pd.read_csv(os.path.join(dir, 'meta_c' + str(batch) + '.csv'), index_col=0)["celltype"].values.squeeze())
        
    try:
        counts_rna = np.array(sp.load_npz(os.path.join(dir, 'GxC' + str(batch) + ".npz")).todense().T)
        counts_rna = utils.preprocess(counts_rna, modality = "RNA", log = False)
    except:
        counts_rna = None
    
    # preprocess the count matrix
    counts_rnas.append(counts_rna)

counts = {"rna":counts_rnas}

# obtain the feature name
genes = pd.read_csv(dir + "gene.csv", header = None).values.squeeze()
feats_name = {"rna": genes}
counts["feats_name"] = feats_name

interacts = None

# x_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(np.concatenate(counts["rna"], axis = 0))
# utils.plot_latent_ext([x_umap[:counts["rna"][0].shape[0], :], x_umap[counts["rna"][0].shape[0]:, :]], annos = labels, mode = "separate", save = None, figsize = (10,15), axis_label = "UMAP")
# utils.plot_latent_ext([x_umap[:counts["rna"][0].shape[0], :], x_umap[counts["rna"][0].shape[0]:, :]], annos = labels, mode = "modality", save = None, figsize = (10,7), axis_label = "UMAP")

# In[]
alpha = [1000, 1, 5]
batchsize = 0.1
run = 0
K = 30
Ns = [K] * 8
N_feat = Ns[0]
interval = 1000
T = 4000
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
for mod in model1.A_assos.keys():
    if mod != "shared":
        print(torch.min(model1.A_assos["shared"] + model1.A_assos[mod]).item())

for mod in model1.A_assos.keys():
    if mod != "shared":
        print(torch.mean(model1.A_assos["shared"] + model1.A_assos[mod]).item())

for mod in model1.A_assos.keys():
    if mod != "shared":
        print(torch.max(model1.A_assos["shared"] + model1.A_assos[mod]).item())

print(model1.scales)

# In[]
umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4, random_state = 0) 
zs = []
for batch in range(8):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)

x_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))
# separate into batches
x_umaps = []
for batch in range(8):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
    elif batch == 7:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = result_dir + f'latent_separate_{K}_{T}.png', figsize = (10,40), axis_label = "UMAP")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = result_dir + f'latent_batches_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = result_dir + f'latent_clusters_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6)

# In[]
# n_neighbors should be larger than the number of batches
n_neighbors = 30

zs = []
for batch in range(8):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)

s_pair_dist, knn_indices, knn_dists = utils.re_nn_distance(zs, n_neighbors)
# s_pair_dist, knn_indices, knn_dists = utils.re_distance_nn(zs, n_neighbors)

umap_op = UMAP(n_components = 2, n_neighbors = n_neighbors, min_dist = 0.1, random_state = 0, 
                metric='precomputed', knn_dists=knn_dists, knn_indices=knn_indices)
x_umap = umap_op.fit_transform(s_pair_dist)
# separate into batches
x_umaps = []
for batch in range(8):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
    elif batch == 7:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = result_dir + f'latent_separate_processed_{K}_{T}.png', 
                      figsize = (10,40), axis_label = "Latent")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = result_dir + f'latent_batches_processed_{K}_{T}.png', 
                      figsize = (15,10), axis_label = "Latent", markerscale = 6)

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = result_dir + f'latent_clusters_processed_{K}_{T}.png', 
                      figsize = (15,10), axis_label = "Latent", markerscale = 6)

# %%
