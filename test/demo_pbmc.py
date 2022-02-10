# In[]
import sys, os
sys.path.append('../')
sys.path.append('../src/')

import numpy as np
from sklearn.decomposition import PCA
from umap_batch import UMAP
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd  
import scipy.sparse as sp

import model
import utils
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# In[]
dir = "../data/real/ASAP-PBMC/"
result_dir = "pbmc/cfrm_quantile/"
seurat_path = "pbmc/seurat/"
liger_path = "pbmc/liger/"

nbatches = 4
counts_rnas = []
counts_atacs = []
counts_proteins = []
labels = []
prec_labels = []
for batch in range(nbatches):
    labels.append(pd.read_csv(os.path.join(dir, 'meta_c' + str(batch + 1) + '.csv'), index_col=0)["coarse_cluster"].values.squeeze())
    prec_labels.append(pd.read_csv(os.path.join(dir, 'meta_c' + str(batch + 1) + '.csv'), index_col=0)["cluster"].values.squeeze())
    try:
        counts_atac = np.array(sp.load_npz(os.path.join(dir, 'RxC' + str(batch + 1) + ".npz")).todense().T)
        counts_atac = utils.preprocess(counts_atac, modality = "ATAC")
    except:
        counts_atac = None
        
    try:
        counts_rna = np.array(sp.load_npz(os.path.join(dir, 'GxC' + str(batch + 1) + ".npz")).todense().T)
        counts_rna = utils.preprocess(counts_rna, modality = "RNA", log = False)
    except:
        counts_rna = None
    
    try:
        # the log transform produce better results for the protein
        counts_protein = np.array(sp.load_npz(os.path.join(dir, 'PxC' + str(batch + 1) + ".npz")).todense().T)
        counts_protein = utils.preprocess(counts_protein, modality = "RNA", log = True)
    except:
        counts_protein = None
    
    # preprocess the count matrix
    counts_rnas.append(counts_rna)
    counts_atacs.append(counts_atac)
    counts_proteins.append(counts_protein)

counts = {"rna":counts_rnas, "atac": counts_atacs, "protein": counts_proteins}

A1 = sp.load_npz(os.path.join(dir, 'GxP.npz'))
A2 = sp.load_npz(os.path.join(dir, 'GxR.npz'))
A1 = np.array(A1.todense())
A2 = np.array(A2.todense())
A1 = utils.preprocess(A1, modality = "interaction")
A2 = utils.preprocess(A2, modality = "interaction")
# No need for pseudo-count matrix
# interacts = {"rna_atac": A2, "rna_protein": A1}
interacts = None

# obtain the feature name
genes = pd.read_csv(dir + "genes.txt", header = None).values.squeeze()
regions = pd.read_csv(dir + "regions.txt", header = None).values.squeeze()
proteins = pd.read_csv(dir + "proteins.txt", header = None).values.squeeze()

feats_name = {"rna": genes, "atac": regions, "protein": proteins}
counts["feats_name"] = feats_name

# In[]
alpha = [1000, 1, 5]
batchsize = 0.1
run = 0
K = 30
Ns = [K] * 4
N_feat = Ns[0]
interval = 1000
T = 4000
lr = 1e-2

start_time = time.time()
model1 = model.cfrm_vanilla(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
losses1 = model1.train_func(T = T)
end_time = time.time()
print("running time: " + str(end_time - start_time))

x = np.linspace(0, T, int(T/interval)+1)
plt.plot(x, losses1)

torch.save(model1.state_dict(), result_dir + f'CFRM_{K}_{T}.pt')
model1.load_state_dict(torch.load(result_dir + f'CFRM_{K}_{T}.pt'))

# In[] Sanity check, the scales should be positive, A_assos should also be positive
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
umap_op = UMAP(n_components = 2, n_neighbors = 30, min_dist = 0.2, random_state = 0) 
zs = []
for batch in range(0,4):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)
    
x_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

# separate into batches
x_umaps = []
leiden_labels = []
for batch in range(nbatches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
        
    elif batch == (nbatches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
        
    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = result_dir + f'latent_separate_{K}_{T}.png', figsize = (10,15), axis_label = "UMAP")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = result_dir + f'latent_batches_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = result_dir + f'latent_clusters_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent_ext(x_umaps, annos = prec_labels, mode = "separate", save = result_dir + f'latent_separate2_{K}_{T}.png', figsize = (10,15), axis_label = "UMAP")

utils.plot_latent_ext(x_umaps, annos = prec_labels, mode = "joint", save = result_dir + f'latent_clusters2_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6)

# In[]
n_neighbors = 30

zs = []
for batch in range(nbatches):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)

s_pair_dist, knn_indices, knn_dists = utils.re_nn_distance(zs, n_neighbors)
# s_pair_dist, knn_indices, knn_dists = re_distance_nn(zs, n_neighbors)
umap_op = UMAP(n_components = 2, n_neighbors = n_neighbors, min_dist = 0.1, random_state = 0, 
                metric='precomputed', knn_dists=knn_dists, knn_indices=knn_indices)
x_umap = umap_op.fit_transform(s_pair_dist)

# separate into batches
x_umaps = []
for batch in range(nbatches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
    elif batch == (nbatches - 1):
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

utils.plot_latent_ext(x_umaps, annos = prec_labels, mode = "separate", save = result_dir + f'latent_separate2_{K}_{T}_processed.png', 
                      figsize = (10,15), axis_label = "Latent")

utils.plot_latent_ext(x_umaps, annos = prec_labels, mode = "joint", save = result_dir + f'latent_clusters2_{K}_{T}_processed.png', 
                      figsize = (15,10), axis_label = "Latent", markerscale = 6)
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
