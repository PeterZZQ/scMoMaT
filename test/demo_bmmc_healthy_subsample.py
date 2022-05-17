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
import umap_batch
from umap import UMAP

import pandas as pd  
import scipy.sparse as sp
import model
import time
import bmk

import quantile 

import coupleNMF as coupleNMF

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# In[]
# read in dataset
dir = '../data/real/diag/healthy_hema/topgenes_1000/BMMC/'
result_dir = "bmmc_healthyhema_1000/remove_celltype/scmomat/"
seurat_path = "bmmc_healthyhema_1000/remove_celltype/seurat/"
liger_path = "bmmc_healthyhema_1000/remove_celltype/liger/"
uinmf_path = "bmmc_healthyhema_1000/remove_celltype/uinmf_bin/" 
multimap_path = "bmmc_healthyhema_1000/remove_celltype/multimap/"


counts_rnas = []
counts_atacs = []
labels = []
n_batches = 2
for batch in range(1, n_batches+1):
    labels.append(pd.read_csv(os.path.join(dir, 'meta_c' + str(batch) + '.csv'), index_col=0)["BioClassification"].values.squeeze())
    
    try:
        counts_atac = np.array(sp.load_npz(os.path.join(dir, 'RxC' + str(batch) + ".npz")).todense().T)
        counts_atac = utils.preprocess(counts_atac, modality = "ATAC")   
    except:
        counts_atac = None
        
    try:
        counts_rna = np.array(sp.load_npz(os.path.join(dir, 'GxC' + str(batch) + ".npz")).todense().T)
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

counts["nbatches"] = n_batches  
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
# model1 = model.cfrm_vanilla(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
model1 = model.cfrm_vanilla(counts = counts, K = K, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run, device = device)
losses1 = model1.train_func(T = T)
end_time = time.time()
print("running time: " + str(end_time - start_time))

x = np.linspace(0, T, int(T/interval) + 1)
plt.plot(x, losses1)

# state dict does not include the scaling factor, etc
# torch.save(model1.state_dict(), result_dir + f'CFRM_{K}_{T}.pt')
# model1.load_state_dict(torch.load(result_dir + f'CFRM_{K}_{T}.pt'))
torch.save(model1, result_dir + f'CFRM_{K}_{T}.pt')
model1 = torch.load(result_dir + f'CFRM_{K}_{T}.pt')

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
for batch in range(0,n_batches):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)

x_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))
# separate into batches
x_umaps = []
for batch in range(0,n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
    elif batch == (n_batches-1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = result_dir + f'latent_separate_{K}_{T}.png', figsize = (10,15), axis_label = "UMAP")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = result_dir + f'latent_batches_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = result_dir + f'latent_clusters_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6)

# In[] Post-processing and clustering
n_neighbors = 30

zs = []
for batch in range(n_batches):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)
# s_pair_dist, knn_indices, knn_dists = utils.re_nn_distance(zs, n_neighbors)
# s_pair_dist, knn_indices, knn_dists = re_distance_nn(zs, n_neighbors)
# s_pair_dist, knn_indices, knn_dists = utils.post_nn_distance(zs, n_neighbors, njobs = 8)
s_pair_dist, knn_indices, knn_dists = utils.post_nn_distance2(zs, n_neighbors, njobs = 8)


# here load the score.csv that we calculated in advance to select the best resolution
# scores = pd.read_csv(result_dir + "score2.csv", index_col = 0)
# scores = scores[scores["methods"] == "scMoMaT"] 
# resolution = scores["resolution"].values[np.argmax(scores["NMI"].values.squeeze())]
# print(resolution)
# resolution = 1.0 for 1000 genes, 2.0 for 2000 genes
resolution = 1.0

labels_tmp = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
umap_op = umap_batch.UMAP(n_components = 2, n_neighbors = n_neighbors, min_dist = 0.2, random_state = 0, 
                metric='precomputed', knn_dists=knn_dists, knn_indices=knn_indices)
x_umap_scmomat = umap_op.fit_transform(s_pair_dist)


# scDART
zs2 = utils.match_embeds(zs, k = n_neighbors, reference = None, bandwidth = 40)
# x_umap = UMAP(n_components = 2, min_dist = 0.2, random_state = 0).fit_transform(np.concatenate(zs2, axis = 0))
# labels_tmp = utils.leiden_cluster(X = np.concatenate(zs2, axis = 0), knn_indices = None, knn_dists = None, resolution = 0.3)


# separate into batches
x_umaps_scmomat = []
leiden_labels = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps_scmomat.append(x_umap_scmomat[start_pointer:end_pointer,:])
        leiden_labels.append(labels_tmp[start_pointer:end_pointer])

    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        x_umaps_scmomat.append(x_umap_scmomat[start_pointer:,:])
        leiden_labels.append(labels_tmp[start_pointer:])

    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps_scmomat.append(x_umap_scmomat[start_pointer:end_pointer,:])
        leiden_labels.append(labels_tmp[start_pointer:end_pointer])

utils.plot_latent_ext(x_umaps_scmomat, annos = labels, mode = "separate", save = result_dir + f'latent_separate_{K}_{T}_processed2.png', 
                      figsize = (15,15), axis_label = "Latent", markerscale = 6, s = 5, label_inplace = True)

utils.plot_latent_ext(x_umaps_scmomat, annos = labels, mode = "modality", save = result_dir + f'latent_batches_{K}_{T}_processed2.png', 
                      figsize = (15,10), axis_label = "Latent", markerscale = 6, s = 5, label_inplace = True)

utils.plot_latent_ext(x_umaps_scmomat, annos = labels, mode = "joint", save = result_dir + f'latent_clusters_{K}_{T}_processed2.png', 
                      figsize = (15,10), axis_label = "Latent", markerscale = 6, s = 5, label_inplace = True)

utils.plot_latent_ext(x_umaps_scmomat, annos = leiden_labels, mode = "joint", save = result_dir + f'latent_leiden_clusters_{K}_{T}_{resolution}_processed2.png', 
                      figsize = (15,10), axis_label = "Latent", markerscale = 6, s = 5, label_inplace = True)


# In[] Baseline methods
# 1. UINMF
H2_uinmf = pd.read_csv(uinmf_path + "liger_c1_norm.csv", index_col = 0).values
H1_uinmf = pd.read_csv(uinmf_path + "liger_c2_norm.csv", index_col = 0).values
uinmf_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(np.concatenate((H1_uinmf, H2_uinmf), axis = 0))
uinmf_umaps = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        uinmf_umaps.append(uinmf_umap[start_pointer:end_pointer,:])
    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        uinmf_umaps.append(uinmf_umap[start_pointer:,:])
    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        uinmf_umaps.append(uinmf_umap[start_pointer:end_pointer,:])

utils.plot_latent_ext(uinmf_umaps, annos = labels, mode = "separate", save = uinmf_path + f'latent_separate_uinmf.png', 
                      figsize = (15,15), axis_label = "Latent", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "large", colormap = "tab20b")

utils.plot_latent_ext(uinmf_umaps, annos = labels, mode = "modality", save = uinmf_path + f'latent_batches_uinmf.png', 
                      figsize = (10,7), axis_label = "Latent", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "large", colormap = "tab20b")

utils.plot_latent_ext(uinmf_umaps, annos = labels, mode = "joint", save = uinmf_path + f'latent_clusters_uinmf.png', 
                      figsize = (10,7), axis_label = "Latent", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "large", colormap = "tab20b")


# 2. Multimap
batches = pd.read_csv(multimap_path + "batch_id.csv", index_col = 0)
X_multimap = np.load(multimap_path + "multimap.npy")
G_multimap = sp.load_npz(multimap_path + "multimap_graph.npz").todense()
X_multimaps = []
for batch in ["C1", "C2"]:
    X_multimaps.append(X_multimap[batches.values.squeeze() == batch, :])

utils.plot_latent_ext(X_multimaps, annos = labels, mode = "separate", save = multimap_path + f'latent_separate_multimap.png', 
                      figsize = (15,15), axis_label = "Latent", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "large", colormap = "tab20b")

utils.plot_latent_ext(X_multimaps, annos = labels, mode = "modality", save = multimap_path + f'latent_batches_multimap.png', 
                      figsize = (10,7), axis_label = "Latent", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "large", colormap = "tab20b")

utils.plot_latent_ext(X_multimaps, annos = labels, mode = "joint", save = multimap_path + f'latent_clusters_multimap.png', 
                      figsize = (10,7), axis_label = "Latent", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "large", colormap = "tab20b")


# 1. Seurat
seurat_pcas = [pd.read_csv(seurat_path + "seurat_pca_c1.txt", sep = "\t", index_col = 0).values, 
               pd.read_csv(seurat_path + "seurat_pca_c2.txt", sep = "\t", index_col = 0).values]
seurat_umaps = [pd.read_csv(seurat_path + "seurat_umap_c1.txt", sep = "\t", index_col = 0).values,
               pd.read_csv(seurat_path + "seurat_umap_c2.txt", sep = "\t", index_col = 0).values]


utils.plot_latent_ext(seurat_umaps, annos = labels, mode = "separate", save = seurat_path + f'latent_separate_seurat.png', 
                      figsize = (15,15), axis_label = "Latent", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "large", colormap = "tab20b")

utils.plot_latent_ext(seurat_umaps, annos = labels, mode = "modality", save = seurat_path + f'latent_batches_seurat.png', 
                      figsize = (10,7), axis_label = "Latent", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "large", colormap = "tab20b")

utils.plot_latent_ext(seurat_umaps, annos = labels, mode = "joint", save = seurat_path + f'latent_clusters_seurat.png', 
                      figsize = (10,7), axis_label = "Latent", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "large", colormap = "tab20b")


# 2. Liger
H1_liger = pd.read_csv(liger_path + "liger_c1.csv", sep = ",", index_col = 0).values
H2_liger = pd.read_csv(liger_path + "liger_c2.csv", sep = ",", index_col = 0).values
liger_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(np.concatenate((H1_liger, H2_liger), axis = 0))
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
                      figsize = (15,15), axis_label = "Latent", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "large", colormap = "Paired")

utils.plot_latent_ext(liger_umaps, annos = labels, mode = "modality", save = liger_path + f'latent_batches_liger.png', 
                      figsize = (10,7), axis_label = "Latent", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "large", colormap = "Paired")

utils.plot_latent_ext(liger_umaps, annos = labels, mode = "joint", save = liger_path + f'latent_clusters_liger.png', 
                      figsize = (10,7), axis_label = "Latent", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "large", colormap = "Paired")




# In[]
importlib.reload(bmk)
# note that here the result change with the number of neighobrs, 30, 20, etc


# labels[1] = np.where(labels[1] == "Memory_CD8_T", "T_CD8_naive", labels[1])
# graph connectivity score (gc) measure the batch effect removal per cell identity
# 1. scMoMaT
# construct neighborhood graph from the post-processed latent space
knn_graph = np.zeros((knn_indices.shape[0], knn_indices.shape[0]))
knn_graph[np.arange(knn_indices.shape[0])[:, None], knn_indices] = 1
gc_scmomat = bmk.graph_connectivity(G = knn_graph, groups = np.concatenate(labels, axis = 0))
print('GC (scmomat): {:.3f}'.format(gc_scmomat))

# 2. Seurat, n_neighbors affect the overall acc, and should be the same as scJMT
n_neighbors = knn_indices.shape[1]
gc_seurat = bmk.graph_connectivity(X = np.concatenate(seurat_pcas, axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (Seurat): {:.3f}'.format(gc_seurat))

# 3. Liger
gc_liger = bmk.graph_connectivity(X = np.concatenate((H1_liger, H2_liger), axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (Liger): {:.3f}'.format(gc_liger))

# 4. UINMF
gc_uinmf = bmk.graph_connectivity(X = np.concatenate((H1_uinmf, H2_uinmf), axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (UINMF): {:.3f}'.format(gc_uinmf))

# 5. Multimap
G_multimap[G_multimap == 0] = np.inf
knn_indices_multimap = G_multimap.argsort(axis = 1)[:, :n_neighbors]
knn_graph_multimap = np.zeros_like(G_multimap)
knn_graph_multimap[np.arange(knn_indices_multimap.shape[0])[:, None], knn_indices_multimap] = 1
gc_multimap = bmk.graph_connectivity(G = knn_graph_multimap, groups = np.concatenate(labels, axis = 0), k = n_neighbors)
gc_multimap2 = bmk.graph_connectivity(X = np.concatenate(X_multimaps, axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (MultiMap Graph): {:.3f}'.format(gc_multimap))
print('GC (MultiMap): {:.3f}'.format(gc_multimap2))

# # 4. scJMT embedding
# gc_scjmt_embed = bmk.graph_connectivity(X = np.concatenate(zs2, axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
# print('GC (scJMT embed): {:.3f}'.format(gc_scjmt_embed))

# Batch effect removal regardless of cell identity
# Graph iLISI

# Conservation of biological identity
# NMI and ARI
# 1. scMoMaT
nmi_scmomat = []
ari_scmomat = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_scjmt = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
    nmi_scmomat.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_scjmt))
    ari_scmomat.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_scjmt))
print('NMI (scMoMaT): {:.3f}'.format(max(nmi_scmomat)))
print('ARI (scMoMaT): {:.3f}'.format(max(ari_scmomat)))

# 2. Seurat
nmi_seurat = []
ari_seurat = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_seurat = utils.leiden_cluster(X = np.concatenate(seurat_pcas, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_seurat.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_seurat))
    ari_seurat.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_seurat))
print('NMI (Seurat): {:.3f}'.format(max(nmi_seurat)))
print('ARI (Seurat): {:.3f}'.format(max(ari_seurat)))

# 3. Liger
nmi_liger = []
ari_liger = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_liger = utils.leiden_cluster(X = np.concatenate((H1_liger, H2_liger), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_liger.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_liger))
    ari_liger.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_liger))
print('NMI (Liger): {:.3f}'.format(max(nmi_liger)))
print('ARI (Liger): {:.3f}'.format(max(ari_liger)))

# 4. UINMF
nmi_uinmf = []
ari_uinmf = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_liger = utils.leiden_cluster(X = np.concatenate((H1_uinmf, H2_uinmf), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_uinmf.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_liger))
    ari_uinmf.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_liger))
print('NMI (UINMF): {:.3f}'.format(max(nmi_uinmf)))
print('ARI (UINMF): {:.3f}'.format(max(ari_uinmf)))

# 5. Multimap
G_multimap = sp.load_npz(multimap_path + "multimap_graph.npz").todense()
nmi_multimap = []
ari_multimap = []
for resolution in np.arange(0.1, 10, 0.5):
    # leiden_labels_seurat = utils.leiden_cluster(X = np.concatenate(seurat_pcas, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    # Multimap state to use graph for clustering
    leiden_labels_multimap = utils.leiden_cluster(affin = G_multimap, resolution = resolution)
    nmi_multimap.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_multimap))
    ari_multimap.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_multimap))
print('NMI (MultiMap): {:.3f}'.format(max(nmi_multimap)))
print('ARI (MultiMap): {:.3f}'.format(max(ari_multimap)))

# # 1. scJMT embedding
# nmi_scjmt_embed = []
# ari_scjmt_embed = []
# for resolution in np.arange(0.1, 10, 0.5):
#     leiden_labels_scjmt_embed = utils.leiden_cluster(X = np.concatenate(zs2, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
#     nmi_scjmt_embed.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_scjmt_embed))
#     ari_scjmt_embed.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_scjmt_embed))
# print('NMI (scJMT embed): {:.3f}'.format(max(nmi_scjmt_embed)))
# print('ARI (scJMT embed): {:.3f}'.format(max(ari_scjmt_embed)))

scores = pd.DataFrame(columns = ["methods", "resolution", "NMI", "ARI", "GC"])
scores["NMI"] = np.array(nmi_scmomat + nmi_seurat + nmi_liger + nmi_uinmf + nmi_multimap)
scores["ARI"] = np.array(ari_scmomat + ari_seurat + ari_liger + ari_uinmf + ari_multimap)
scores["GC"] = np.array([gc_scmomat] * len(nmi_scmomat) + [gc_seurat] * len(nmi_seurat) + [gc_liger] * len(nmi_liger) + [gc_uinmf] * len(nmi_uinmf) + [gc_multimap] * len(nmi_multimap))
scores["resolution"] = np.array([x for x in np.arange(0.1, 10, 0.5)] * 5)
scores["methods"] = np.array(["scMoMaT"] * len(nmi_scmomat) + ["Seurat"] * len(nmi_seurat) + ["Liger"] * len(nmi_liger) + ["UINMF"] * len(nmi_uinmf) + ["MultiMap"] * len(nmi_multimap))
scores.to_csv(result_dir + "score2.csv")


# In[]
# # score for the original post-processing
# scores = pd.read_csv(result_dir + "score.csv")
# # score for post_nn_distance
# scores1 = pd.read_csv(result_dir + "score1.csv")
# score for post_nn_distance2
scores2 = pd.read_csv(result_dir + "score2.csv", index_col = 0)
# print("GC (scMoMaT) postprocess ori: {:.4f}".format(np.max(scores.loc[scores["methods"] == "scMoMaT", "GC"].values)))
# print("NMI (scMoMaT) postprocess ori: {:.4f}".format(np.max(scores.loc[scores["methods"] == "scMoMaT", "NMI"].values)))
# print("ARI (scMoMaT) postprocess ori: {:.4f}".format(np.max(scores.loc[scores["methods"] == "scMoMaT", "ARI"].values)))

# print("GC (scMoMaT) postprocess 1: {:.4f}".format(np.max(scores1.loc[scores["methods"] == "scMoMaT", "GC"].values)))
# print("NMI (scMoMaT) postprocess 1: {:.4f}".format(np.max(scores1.loc[scores["methods"] == "scMoMaT", "NMI"].values)))
# print("ARI (scMoMaT) postprocess 1: {:.4f}".format(np.max(scores1.loc[scores["methods"] == "scMoMaT", "ARI"].values)))

print("GC (scMoMaT) postprocess 2: {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "scMoMaT", "GC"].values)))
print("NMI (scMoMaT) postprocess 2: {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "scMoMaT", "NMI"].values)))
print("ARI (scMoMaT) postprocess 2: {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "scMoMaT", "ARI"].values)))

print("GC (UINMF): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "UINMF", "GC"].values)))
print("NMI (UINMF): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "UINMF", "NMI"].values)))
print("ARI (UINMF): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "UINMF", "ARI"].values)))

print("GC (MultiMap): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "MultiMap", "GC"].values)))
print("NMI (MultiMap): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "MultiMap", "NMI"].values)))
print("ARI (MultiMap): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "MultiMap", "ARI"].values)))

print("GC (LIGER): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Liger", "GC"].values)))
print("NMI (LIGER): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Liger", "NMI"].values)))
print("ARI (LIGER): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Liger", "ARI"].values)))

print("GC (Seurat): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Seurat", "GC"].values)))
print("NMI (Seurat): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Seurat", "NMI"].values)))
print("ARI (Seurat): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Seurat", "ARI"].values)))

# performance not good
# scores2 = pd.read_csv(result_dir + "score2_2.csv", index_col = 0)
# print("GC (scMoMaT) postprocess 2: {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "scMoMaT", "GC"].values)))
# print("NMI (scMoMaT) postprocess 2: {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "scMoMaT", "NMI"].values)))
# print("ARI (scMoMaT) postprocess 2: {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "scMoMaT", "ARI"].values)))

# print("GC (UINMF): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "UINMF", "GC"].values)))
# print("NMI (UINMF): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "UINMF", "NMI"].values)))
# print("ARI (UINMF): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "UINMF", "ARI"].values)))

# print("GC (MultiMap): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "MultiMap", "GC"].values)))
# print("NMI (MultiMap): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "MultiMap", "NMI"].values)))
# print("ARI (MultiMap): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "MultiMap", "ARI"].values)))

# print("GC (LIGER): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Liger", "GC"].values)))
# print("NMI (LIGER): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Liger", "NMI"].values)))
# print("ARI (LIGER): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Liger", "ARI"].values)))

# print("GC (Seurat): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Seurat", "GC"].values)))
# print("NMI (Seurat): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Seurat", "NMI"].values)))
# print("ARI (Seurat): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Seurat", "ARI"].values)))
