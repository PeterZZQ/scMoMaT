# In[]
import sys, os
sys.path.append('../')
sys.path.append('../src/')

import torch
import numpy as np
import utils
from torch.nn import Module, Parameter
import torch.optim as opt
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
subsampling = 1
dir = '../data/real/diag/healthy_hema/topgenes_1000/BMMC/'
if subsampling == 1:
    result_dir = "bmmc_healthyhema_1000/cfrm_quantile/"
    seurat_path = "bmmc_healthyhema_1000/seurat/"
    liger_path = "bmmc_healthyhema_1000/liger/"
else:
    result_dir = "bmmc_healthyhema_1000/subsample_" + str(subsampling) + "/cfrm_quantile/"
    seurat_path = "bmmc_healthyhema_1000/subsample_" + str(subsampling) + "/seurat/"
    liger_path = "bmmc_healthyhema_1000/subsample_" + str(subsampling) + "/liger/"

counts_rnas = []
counts_atacs = []
labels = []
for batch in [1, 2]:
    labels.append(pd.read_csv(os.path.join(dir, 'meta_c' + str(batch) + '.csv'), index_col=0)["BioClassification"].values.squeeze()[::subsampling])
    
    try:
        counts_atac = np.array(sp.load_npz(os.path.join(dir, 'RxC' + str(batch) + ".npz")).todense().T)[::subsampling,:]
        counts_atac = utils.preprocess(counts_atac, modality = "ATAC")   
    except:
        counts_atac = None
        
    try:
        counts_rna = np.array(sp.load_npz(os.path.join(dir, 'GxC' + str(batch) + ".npz")).todense().T)[::subsampling,:]
        counts_rna = utils.preprocess(counts_rna, modality = "RNA", log = False)
    except:
        counts_rna = None
    
    counts_rnas.append(counts_rna)
    counts_atacs.append(counts_atac)

counts = {"rna":counts_rnas, "atac": counts_atacs}

A = sp.load_npz(os.path.join(dir, 'GxR.npz'))
A = np.array(A.todense())
interacts = None

genes = pd.read_csv(dir + "genes.txt", header = None).values.squeeze()
regions = pd.read_csv(dir + "regions.txt", header = None).values.squeeze()

feats_name = {"rna": genes, "atac": regions}
counts["feats_name"] = feats_name

# CALCULATE PSEUDO-SCRNA-SEQ
counts["rna"][0] = counts["atac"][0] @ A.T
#BINARIZE, still is able to see the cluster pattern
counts["rna"][0] = (counts["rna"][0]!=0).astype(int)

# PLOT FUNCTION
# x_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(np.concatenate(counts["rna"], axis = 0))
# utils.plot_latent_ext([x_umap[:counts["rna"][0].shape[0], :], x_umap[counts["rna"][0].shape[0]:, :]], annos = labels, mode = "separate", save = None, figsize = (10,15), axis_label = "UMAP")
# utils.plot_latent_ext([x_umap[:counts["rna"][0].shape[0], :], x_umap[counts["rna"][0].shape[0]:, :]], annos = labels, mode = "modality", save = None, figsize = (10,7), axis_label = "UMAP")

# In[]
alpha = [1000, 1, 5]
batchsize = 0.1
run = 0
K = 30
Ns = [K] * 2
N_feat = Ns[0]
interval = 1000
T = 4000
lr = 1e-2

start_time = time.time()
model1 = model.cfrm_vanilla(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
losses1 = model1.train_func(T = T)
end_time = time.time()
print("running time: " + str(end_time - start_time))

x = np.linspace(0, T, int(T/interval) + 1)
plt.plot(x, losses1)

torch.save(model1.state_dict(), result_dir + f'CFRM_{K}_{T}.pt')
model1.load_state_dict(torch.load(result_dir + f'CFRM_{K}_{T}.pt'))

# In[] Check the scales is positive
for mod in model1.A_assos.keys():
    if mod != "shared":
        print("minimum")
        print(mod)
        print(torch.min(model1.A_assos["shared"] + model1.A_assos[mod]).item())

for mod in model1.A_assos.keys():
    if mod != "shared":
        print("mean")
        print(mod)
        print(torch.mean(model1.A_assos["shared"] + model1.A_assos[mod]).item())

for mod in model1.A_assos.keys():
    if mod != "shared":
        print("maximum")
        print(mod)
        print(torch.max(model1.A_assos["shared"] + model1.A_assos[mod]).item())

print(model1.scales)
# In[]
umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4, random_state = 0) 
zs = []
for batch in range(0,2):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)

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
n_neighbors = 30

zs = []
for batch in range(0,2):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)

s_pair_dist, knn_indices, knn_dists = utils.re_nn_distance(zs, n_neighbors)
# s_pair_dist, knn_indices, knn_dists = re_distance_nn(zs, n_neighbors)
umap_op = UMAP(n_components = 2, n_neighbors = n_neighbors, min_dist = 0.2, random_state = 0, 
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

utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = result_dir + f'latent_separate_{K}_{T}_processed.png', 
                      figsize = (10,15), axis_label = "Latent")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = result_dir + f'latent_batches_{K}_{T}_processed.png', 
                      figsize = (15,10), axis_label = "Latent", markerscale = 6)

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = result_dir + f'latent_clusters_{K}_{T}_processed.png', 
                      figsize = (15,10), axis_label = "Latent", markerscale = 6)

# In[] Baseline methods
'''
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
H1 = pd.read_csv(liger_path + "liger_c1.csv", sep = ",", index_col = 0).values
H2 = pd.read_csv(liger_path + "liger_c2.csv", sep = ",", index_col = 0).values
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
'''
# %%
