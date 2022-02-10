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
# In[] Using pseudo-scRNA-Seq 
subsampling = 1
dir = '../data/real/diag/healthy_hema/topgenes_1000/BMMC/'
if subsampling == 1:
    result_dir = "bmmc_healthyhema_1000/cfrm_quantile/"
else:
    result_dir = "bmmc_healthyhema_1000/subsample_" + str(subsampling) + "/cfrm_quantile/"

counts_rnas = []
counts_atacs = []
counts_proteins = []
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
        # counts_rna = utils.preprocess_liger(counts_rna, with_mean = False)
        # counts_rna = (counts_rna!=0).astype(int)
        # print(np.sum(counts_rna)/(counts_rna.shape[0] * counts_rna.shape[1]))
    except:
        counts_rna = None
    
    counts_rnas.append(counts_rna)
    counts_atacs.append(counts_atac)

counts = {"rna":counts_rnas, "atac": counts_atacs}

A = sp.load_npz(os.path.join(dir, 'GxR.npz'))
A = np.array(A.todense())
# normalize A
# A = utils.preprocess(A, modality = "interaction")
interacts = None

genes = pd.read_csv(dir + "genes.txt", header = None).values.squeeze()
regions = pd.read_csv(dir + "regions.txt", header = None).values.squeeze()

feats_name = {"rna": genes, "atac": regions}
counts["feats_name"] = feats_name

# CALCULATE PSEUDO-SCRNA-SEQ
counts["rna"][0] = counts["atac"][0] @ A.T
#BINARIZE, still is able to see the cluster pattern
counts["rna"][0] = (counts["rna"][0]!=0).astype(int)
print(np.sum(counts["rna"][0])/(counts["rna"][0].shape[0] * counts["rna"][0].shape[1]))
# NORMALIZATION OF PSEUDO-SCRNA-SEQ
# counts["rna"][0] = utils.preprocess(counts["rna"][0], modality = "RNA", log = True)
# LIGER flavor
# counts["rna"][0] = utils.preprocess_liger(counts["rna"][0], with_mean = False)
# quantile
# counts["rna"][0] = quantile_norm(targ_mtx = counts["rna"][0], ref_mtx = counts["rna"][1], replace = True)

# PLOT FUNCTION
x_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(np.concatenate(counts["rna"], axis = 0))
utils.plot_latent_ext([x_umap[:counts["rna"][0].shape[0], :], x_umap[counts["rna"][0].shape[0]:, :]], annos = labels, mode = "separate", save = None, figsize = (10,15), axis_label = "UMAP")
utils.plot_latent_ext([x_umap[:counts["rna"][0].shape[0], :], x_umap[counts["rna"][0].shape[0]:, :]], annos = labels, mode = "modality", save = None, figsize = (10,7), axis_label = "UMAP")
# counts = {"rna": counts["rna"], "feats_name": feats_name}

# In[] Using interaction
subsampling = 3
dir = '../data/real/diag/healthy_hema/topgenes_1000/BMMC/'
if subsampling == 1:
    result_dir = "bmmc_healthyhema_1000/cfrm_quantile/"
else:
    result_dir = "bmmc_healthyhema_1000/subsample_" + str(subsampling) + "/cfrm_quantile/"

counts_rnas = []
counts_atacs = []
counts_proteins = []
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
        counts_rna = utils.preprocess(counts_rna, modality = "RNA", log = True)
        # counts_rna = utils.preprocess_liger(counts_rna, with_mean = False)
    except:
        counts_rna = None
    
    counts_rnas.append(counts_rna)
    counts_atacs.append(counts_atac)

counts = {"rna":counts_rnas, "atac": counts_atacs}

A = sp.load_npz(os.path.join(dir, 'GxR.npz'))
A = np.array(A.todense())
# normalize A
A = utils.preprocess(A, modality = "interaction")
interacts = {"rna_atac": A}

genes = pd.read_csv(dir + "genes.txt", header = None).values.squeeze()
regions = pd.read_csv(dir + "regions.txt", header = None).values.squeeze()

feats_name = {"rna": genes, "atac": regions}
counts["feats_name"] = feats_name

# In[]
#hyper parameters: best lr = 5e-3, T = 4000, latent_dims = 13
alpha = [1000, 1, 5]
batchsize = 0.1
run = 0
K = 30
Ns = [K] * 2
N_feat = Ns[0]
interval = 1000
T = 4000
lr = 1e-2
model1 = model.cfrm_vanilla(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
losses1 = model1.train_func(T = T)
# torch.save(model1.state_dict(), result_dir + f'CFRM_{K}_{T}.pt')
# model1.load_state_dict(torch.load(result_dir + f'CFRM_{K}_{T}.pt'))

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

utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = None, figsize = (10,15), axis_label = "UMAP")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = None, figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = None, figsize = (15,10), axis_label = "UMAP", markerscale = 6)

# In[]
n_neighbors = 30

zs = []
for batch in range(0,2):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    # z = model1.C_cells[batch].cpu().detach().numpy()
    zs.append(z)

s_pair_dist, knn_indices, knn_dists = utils.re_nn_distance(zs, n_neighbors)
# s_pair_dist, knn_indices, knn_dists = utils.re_distance_nn(zs, n_neighbors)

umap_op = UMAP(n_components = 2, n_neighbors = n_neighbors, min_dist = 0.2, random_state = 0, 
                metric='precomputed', knn_dists=knn_dists, knn_indices=knn_indices)
x_umap = umap_op.fit_transform(s_pair_dist)

# zs2 = utils.match_embeds(zs, k = 2, reference = None, bandwidth = 40)
# x_umap = umap_op.fit_transform(np.concatenate(zs2, axis = 0))
# zs2, _ = quantile.quantile_norm(zs)
# x_umap = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4, random_state = 0).fit_transform(np.concatenate(zs2, axis = 0))

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
# %%
