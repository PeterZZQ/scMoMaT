# In[]
import sys, os
sys.path.append('../')
sys.path.append('../src/')

import numpy as np
import umap_batch
from umap import UMAP
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd  
import scipy.sparse as sp

import model
import utils
import bmk

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.size"] = 10

# In[]
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   NOTE: 1. Load dataset and running scmomat (without retraining, retraining see the third section)
#
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# NOTE: read in dataset
dir = "../data/real/hori/Pancreas/"
result_dir = "pancreas/scmomat/"
seurat_path = "pancreas/seurat/"
liger_path = "pancreas/liger/"

counts_rnas = []
labels = []
n_batches = 8
for batch in range(n_batches):
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
counts["nbatches"] = n_batches
x_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(np.concatenate(counts["rna"], axis = 0))
x_umaps = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + counts_rnas[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
        
    elif batch == (n_batches - 1):
        start_pointer = start_pointer + counts_rnas[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
        
    else:
        start_pointer = start_pointer + counts_rnas[batch - 1].shape[0]
        end_pointer = start_pointer + counts_rnas[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
        
utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = "pancreas/cell_types.png", figsize = (10,7), axis_label = "UMAP")
utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = "pancreas/batches.png", figsize = (10,7), axis_label = "UMAP")


# In[]
# NOTE: Running scmomat
# weight on regularization term
lamb = 0.001
batchsize = 0.1
# running seed
seed = 0
# number of latent dimensions
K = 30
interval = 1000
T = 4000
lr = 1e-2

start_time = time.time()
model1 = model.scmomat(counts = counts, K = K, batch_size = batchsize, interval = interval, lr = lr, lamb = lamb, seed = seed, device = device)
losses1 = model1.train_func(T = T)
end_time = time.time()
print("running time: " + str(end_time - start_time))

# x = np.linspace(0, T, int(T/interval)+1)
# plt.plot(x, losses1)
# plt.yscale("log")

torch.save(model1, result_dir + f'CFRM_{K}_{T}.pt')
model1 = torch.load(result_dir + f'CFRM_{K}_{T}.pt')

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
# NOTE: Plot the result before post-processing (no post-processing for pbmc)
umap_op = UMAP(n_components = 2, n_neighbors = 30, min_dist = 0.2, random_state = 0) 
zs = []
for batch in range(n_batches):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)
    
x_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

# scores = pd.read_csv(result_dir + "score.csv", index_col = 0)
# scores = scores[scores["methods"] == "scMoMaT"] 
# resolution = scores["resolution"].values[np.argmax(scores["NMI (prec)"].values.squeeze())]
# print(resolution)
# resolution = 1
resolution = 0.4
labels_tmp = utils.leiden_cluster(X = np.concatenate(zs, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)

# separate into batches
x_umaps_scmomat = []
leiden_labels = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps_scmomat.append(x_umap[start_pointer:end_pointer,:])
        leiden_labels.append(labels_tmp[start_pointer:end_pointer])
        
    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        x_umaps_scmomat.append(x_umap[start_pointer:,:])
        leiden_labels.append(labels_tmp[start_pointer:])
        
    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps_scmomat.append(x_umap[start_pointer:end_pointer,:])
        leiden_labels.append(labels_tmp[start_pointer:end_pointer])

# utils.plot_latent_ext(x_umaps_scmomat, annos = labels, mode = "separate", save = result_dir + f'latent_separate_{K}_{T}.png', figsize = (10,50), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "x-large")

utils.plot_latent_ext(x_umaps_scmomat, annos = labels, mode = "modality", save = result_dir + f'latent_batches_{K}_{T}.png', figsize = (10,7), axis_label = "UMAP", markerscale = 10, s = 2, label_inplace = True, text_size = "x-large", alpha = 0.7)

utils.plot_latent_ext(x_umaps_scmomat, annos = labels, mode = "joint", save = result_dir + f'latent_clusters_{K}_{T}.png', figsize = (10,7), axis_label = "UMAP", markerscale = 10, s = 2, label_inplace = True, text_size = "x-large", alpha = 0.7, colormap = "Paired")

utils.plot_latent_ext(x_umaps_scmomat, annos = leiden_labels, mode = "joint", save = result_dir + f'latent_leiden_clusters_{K}_{T}_{resolution}.png', figsize = (8,5), axis_label = "UMAP", markerscale = 10, s = 2, label_inplace = True, text_size = "xx-large", alpha = 0.7)


# In[]
# NOTE: Post-processing, clustering, and plot the result after post-processing
n_neighbors = 30
r = 0.9

zs = []
for batch in range(n_batches):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)

s_pair_dist, knn_indices, knn_dists = utils.post_process(zs, n_neighbors, njobs = 8, r = r)

resolution = 0.5
labels_tmp = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
umap_op = umap_batch.UMAP(n_components = 2, n_neighbors = n_neighbors, min_dist = 0.2, random_state = 0, 
                metric='precomputed', knn_dists=knn_dists, knn_indices=knn_indices)
x_umap = umap_op.fit_transform(s_pair_dist)

# separate into batches
x_umaps_scmomat = []
leiden_labels = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps_scmomat.append(x_umap[start_pointer:end_pointer,:])
        leiden_labels.append(labels_tmp[start_pointer:end_pointer])
        
    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        x_umaps_scmomat.append(x_umap[start_pointer:,:])
        leiden_labels.append(labels_tmp[start_pointer:])
        
    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps_scmomat.append(x_umap[start_pointer:end_pointer,:])
        leiden_labels.append(labels_tmp[start_pointer:end_pointer])

# utils.plot_latent_ext(x_umaps_scmomat, annos = labels, mode = "separate", save = result_dir + f'latent_separate_{K}_{T}_postprocessed.png', figsize = (10,20), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "x-large")

utils.plot_latent_ext(x_umaps_scmomat, annos = labels, mode = "modality", save = result_dir + f'latent_batches_{K}_{T}_postprocessed.png', figsize = (8,5), axis_label = "UMAP", markerscale = 10, s = 2, label_inplace = True, text_size = "x-large", alpha = 0.7)

utils.plot_latent_ext(x_umaps_scmomat, annos = labels, mode = "joint", save = result_dir + f'latent_clusters_{K}_{T}_postprocessed.png', figsize = (8,5), axis_label = "UMAP", markerscale = 10, s = 2, label_inplace = True, text_size = "x-large", alpha = 0.7, colormap = "Paired")

utils.plot_latent_ext(x_umaps_scmomat, annos = leiden_labels, mode = "joint", save = result_dir + f'latent_leiden_clusters_{K}_{T}_{resolution}_postprocessed.png', figsize = (8,5), axis_label = "UMAP", markerscale = 10, s = 2, label_inplace = True, text_size = "xx-large", alpha = 0.7)

# In[] 
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   NOTE: 2. Benchmarking with baseline methods
#
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# NOTE: Baseline methods
# UINMF
uinmf_path = "pancreas/LIGER/" 
H0 = pd.read_csv(uinmf_path + "H0_norm.csv", index_col = 0).values
H1 = pd.read_csv(uinmf_path + "H1_norm.csv", index_col = 0).values
H2 = pd.read_csv(uinmf_path + "H2_norm.csv", index_col = 0).values
H3 = pd.read_csv(uinmf_path + "H3_norm.csv", index_col = 0).values
H4 = pd.read_csv(uinmf_path + "H4_norm.csv", index_col = 0).values
H5 = pd.read_csv(uinmf_path + "H5_norm.csv", index_col = 0).values
H6 = pd.read_csv(uinmf_path + "H6_norm.csv", index_col = 0).values
H7 = pd.read_csv(uinmf_path + "H7_norm.csv", index_col = 0).values
liger_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(np.concatenate((H0, H1, H2, H3, H4, H5, H6, H7), axis = 0))
liger_umaps = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        liger_umaps.append(liger_umap[start_pointer:end_pointer,:])
    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        liger_umaps.append(liger_umap[start_pointer:,:])
    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        liger_umaps.append(liger_umap[start_pointer:end_pointer,:])

utils.plot_latent_ext(liger_umaps, annos = labels, mode = "joint", save = uinmf_path + f'latent_uinmf.png', 
                      figsize = (10,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")
utils.plot_latent_ext(liger_umaps, annos = labels, mode = "modality", save = uinmf_path + f'latent_uinmf_batch.png', 
                      figsize = (10,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")

# Seurat
seurat_path = "pancreas/Seurat/"
pca_seurat0 = pd.read_csv(seurat_path + "seurat_pca0.txt", sep = "\t").values
pca_seurat1 = pd.read_csv(seurat_path + "seurat_pca1.txt", sep = "\t").values
pca_seurat2 = pd.read_csv(seurat_path + "seurat_pca2.txt", sep = "\t").values
pca_seurat3 = pd.read_csv(seurat_path + "seurat_pca3.txt", sep = "\t").values
pca_seurat4 = pd.read_csv(seurat_path + "seurat_pca4.txt", sep = "\t").values
pca_seurat5 = pd.read_csv(seurat_path + "seurat_pca5.txt", sep = "\t").values
pca_seurat6 = pd.read_csv(seurat_path + "seurat_pca6.txt", sep = "\t").values
pca_seurat7 = pd.read_csv(seurat_path + "seurat_pca7.txt", sep = "\t").values

umap_seurat0 = pd.read_csv(seurat_path + "seurat_umap0.txt", sep = "\t").values
umap_seurat1 = pd.read_csv(seurat_path + "seurat_umap1.txt", sep = "\t").values
umap_seurat2 = pd.read_csv(seurat_path + "seurat_umap2.txt", sep = "\t").values
umap_seurat3 = pd.read_csv(seurat_path + "seurat_umap3.txt", sep = "\t").values
umap_seurat4 = pd.read_csv(seurat_path + "seurat_umap4.txt", sep = "\t").values
umap_seurat5 = pd.read_csv(seurat_path + "seurat_umap5.txt", sep = "\t").values
umap_seurat6 = pd.read_csv(seurat_path + "seurat_umap6.txt", sep = "\t").values
umap_seurat7 = pd.read_csv(seurat_path + "seurat_umap7.txt", sep = "\t").values

utils.plot_latent_ext([umap_seurat0, umap_seurat1, umap_seurat2, umap_seurat3, umap_seurat4, umap_seurat5, umap_seurat6, umap_seurat7], annos = labels, mode = "joint", save = seurat_path + f'latent_seurat.png', 
                      figsize = (10,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")
utils.plot_latent_ext([umap_seurat0, umap_seurat1, umap_seurat2, umap_seurat3, umap_seurat4, umap_seurat5, umap_seurat6, umap_seurat7], annos = labels, mode = "modality", save = seurat_path + f'latent_seurat_batch.png', 
                      figsize = (10,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")


# Multimap
multimap_path = "pancreas/multimap/"
batches = pd.read_csv(multimap_path + "batch_id.csv", index_col = 0)
X_multimap = np.load(multimap_path + "multimap.npy")
G_multimap = sp.load_npz(multimap_path + "multimap_graph.npz").toarray()
X_multimaps = []
for batch in ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
    X_multimaps.append(X_multimap[batches.values.squeeze() == batch, :])

utils.plot_latent_ext(X_multimaps, annos = labels, mode = "joint", save = multimap_path + f'latent_multimap.png', 
                      figsize = (10,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")
utils.plot_latent_ext(X_multimaps, annos = labels, mode = "modality", save = multimap_path + f'latent_multimap_batch.png', 
                      figsize = (10,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")


# In[]
# n_neighbors = 70
n_neighbors = knn_indices.shape[1]
# graph connectivity score (gc) measure the batch effect removal per cell identity
# 1. scMoMaT
gc_scmomat = bmk.graph_connectivity(X = np.concatenate(zs, axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (scMoMaT): {:.3f}'.format(gc_scmomat))

# knn_graph = np.zeros((knn_indices.shape[0], knn_indices.shape[0]))
# knn_graph[np.arange(knn_indices.shape[0])[:, None], knn_indices] = 1
# gc_scmomat = bmk.graph_connectivity(G = knn_graph, groups = np.concatenate(labels, axis = 0))
# print('GC (scmomat post-processed): {:.3f}'.format(gc_scmomat))


# knn_graph = np.zeros((knn_indices.shape[0], knn_indices.shape[0]))
# knn_graph[np.arange(knn_indices.shape[0])[:, None], knn_indices] = 1
# gc_scmomat = bmk.graph_connectivity(G = knn_graph, groups = np.concatenate(labels, axis = 0))
# print('GC (scmomat): {:.3f}'.format(gc_scmomat))

# 2. UINMF
gc_uinmf = bmk.graph_connectivity(X = np.concatenate((H0, H1, H2, H3, H4, H5, H6, H7), axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (UINMF): {:.3f}'.format(gc_uinmf))

# 3. Seurat
gc_seurat = bmk.graph_connectivity(X = np.concatenate((pca_seurat0, pca_seurat1, pca_seurat2, pca_seurat3, pca_seurat4, pca_seurat5, pca_seurat6, pca_seurat7), axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (Seurat): {:.3f}'.format(gc_seurat))

# 3. Multimap
# NOTE: G_multimap is an affinity graph, closer neighbor with larger value
# argsort from small to large, select the last n_neighbors
G_multimap = sp.load_npz(multimap_path + "multimap_graph.npz").toarray()
knn_indices_multimap = G_multimap.argsort(axis = 1)[:, -n_neighbors:]
knn_graph_multimap = np.zeros_like(G_multimap)
knn_graph_multimap[np.arange(knn_indices_multimap.shape[0])[:, None], knn_indices_multimap] = 1
gc_multimap = bmk.graph_connectivity(G = knn_graph_multimap, groups = np.concatenate(labels, axis = 0), k = n_neighbors)
gc_multimap2 = bmk.graph_connectivity(X = np.concatenate(X_multimaps, axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (MultiMap Graph): {:.3f}'.format(gc_multimap))
print('GC (MultiMap): {:.3f}'.format(gc_multimap2))
# Batch effect removal regardless of cell identity
# Graph iLISI

# Conservation of biological identity
# NMI and ARI
# 1. scMoMaT
nmi_scmomat = []
ari_scmomat = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_scmomat = utils.leiden_cluster(X = np.concatenate(zs, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_scmomat.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_scmomat))
    ari_scmomat.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_scmomat))
print('NMI (scMoMaT): {:.3f}'.format(max(nmi_scmomat)))
print('ARI (scMoMaT): {:.3f}'.format(max(ari_scmomat)))


# nmi_scmomat = []
# ari_scmomat = []
# for resolution in np.arange(0.1, 10, 0.5):
#     leiden_labels_scmomat = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
#     nmi_scmomat.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_scmomat))
#     ari_scmomat.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_scmomat))
# print('NMI (scMoMaT post-processed): {:.3f}'.format(max(nmi_scmomat)))
# print('ARI (scMoMaT post-processed): {:.3f}'.format(max(ari_scmomat)))


# 2. UINMF
nmi_uinmf = []
ari_uinmf = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_uinmf = utils.leiden_cluster(X = np.concatenate((H0, H1, H2, H3, H4, H5, H6, H7), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_uinmf.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_uinmf))
    ari_uinmf.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_uinmf))
print('NMI (UINMF): {:.3f}'.format(max(nmi_uinmf)))
print('ARI (UINMF): {:.3f}'.format(max(ari_uinmf)))

# 3. Seurat
nmi_seurat = []
ari_seurat = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_seurat = utils.leiden_cluster(X = np.concatenate((pca_seurat0, pca_seurat1, pca_seurat2, pca_seurat3, pca_seurat4, pca_seurat5, pca_seurat6, pca_seurat7), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_seurat.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_seurat))
    ari_seurat.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_seurat))
print('NMI (Seurat): {:.3f}'.format(max(nmi_seurat)))
print('ARI (Seurat): {:.3f}'.format(max(ari_seurat)))

# 4. Multimap
G_multimap = sp.load_npz(multimap_path + "multimap_graph.npz").toarray()
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

# Label transfer accuracy
# randomly select a half of cells as query
np.random.seed(0)
query_cell = np.array([False] * G_multimap.shape[0])
query_cell[np.random.choice(np.arange(G_multimap.shape[0]), size = int(0.5 * G_multimap.shape[0]), replace = False)] = True
training_cell = (1 - query_cell).astype(np.bool)
query_label = np.concatenate(labels)[query_cell]
training_label = np.concatenate(labels)[training_cell]

# scmomat
lta_scmomat = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, 
                                    z_query = np.concatenate(zs, axis = 0)[query_cell,:],
                                    z_train = np.concatenate(zs, axis = 0)[training_cell,:])
# knn_graph = np.zeros((knn_indices.shape[0], knn_indices.shape[0]))
# knn_graph[np.arange(knn_indices.shape[0])[:, None], knn_indices] = 1
# knn_graph = knn_graph[query_cell, :][:, training_cell]
# lta_scmomat = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, knn_graph = knn_graph)

# UINMF
lta_uinmf = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, 
                                  z_query = np.concatenate((H0, H1, H2, H3, H4, H5, H6, H7), axis = 0)[query_cell,:],
                                  z_train = np.concatenate((H0, H1, H2, H3, H4, H5, H6, H7), axis = 0)[training_cell,:])

# seurat
lta_seurat = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, 
                                  z_query = np.concatenate((pca_seurat0, pca_seurat1, pca_seurat2, pca_seurat3, pca_seurat4, pca_seurat5, pca_seurat6, pca_seurat7), axis = 0)[query_cell,:],
                                  z_train = np.concatenate((pca_seurat0, pca_seurat1, pca_seurat2, pca_seurat3, pca_seurat4, pca_seurat5, pca_seurat6, pca_seurat7), axis = 0)[training_cell,:])

# MultiMap
G_multimap = sp.load_npz(multimap_path + "multimap_graph.npz").toarray()
knn_indices_multimap = G_multimap.argsort(axis = 1)[:, -n_neighbors:]
knn_graph_multimap = np.zeros_like(G_multimap)
knn_graph_multimap[np.arange(knn_indices_multimap.shape[0])[:, None], knn_indices_multimap] = 1
knn_graph_multimap = knn_graph_multimap[query_cell, :][:, training_cell]
lta_multimap = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, knn_graph = knn_graph_multimap)
lt2_multimap2 = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, 
                                  z_query = np.concatenate(X_multimaps, axis = 0)[query_cell,:],
                                  z_train = np.concatenate(X_multimaps, axis = 0)[training_cell,:])

print("Label transfer accuracy (scMoMaT): {:.3f}".format(lta_scmomat))
print("Label transfer accuracy (UINMF): {:.3f}".format(lta_uinmf))
print("Label transfer accuracy (Seurat): {:.3f}".format(lta_seurat))
print("Label transfer accuracy (MultiMap Graph): {:.3f}".format(lta_multimap))
print("Label transfer accuracy (MultiMap): {:.3f}".format(lt2_multimap2))

scores = pd.DataFrame(columns = ["methods", "resolution", "NMI", "ARI", "GC", "LTA"])
scores["NMI"] = np.array(nmi_scmomat + nmi_uinmf + nmi_multimap)
scores["ARI"] = np.array(ari_scmomat + ari_uinmf + ari_multimap)
scores["GC"] = np.array([gc_scmomat] * len(nmi_scmomat) + [gc_uinmf] * len(nmi_uinmf) + [gc_multimap] * len(ari_multimap))
scores["LTA"] = np.array([lta_scmomat] * len(nmi_scmomat) + [lta_uinmf] * len(nmi_uinmf) + [lta_multimap] * len(ari_multimap))
scores["resolution"] = np.array([x for x in np.arange(0.1, 10, 0.5)] * 3)
scores["methods"] = np.array(["scMoMaT"] * len(nmi_scmomat) + ["UINMF"] * len(nmi_uinmf) + ["MultiMap"] * len(ari_multimap))

scores.to_csv(result_dir + "score.csv")

# In[]
# GC
plt.rcParams["font.size"] = 20
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

score = pd.read_csv(result_dir + "score.csv")
gc_scmomat = np.max(score.loc[score["methods"] == "scMoMaT", "GC"].values)
gc_uinmf = np.max(score.loc[score["methods"] == "UINMF", "GC"].values)
gc_multimap = np.max(score.loc[score["methods"] == "MultiMap", "GC"].values)

fig = plt.figure(figsize = (4,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3], [gc_scmomat, gc_uinmf, gc_multimap], width = 0.4)
barlist[0].set_color('r')
fig.savefig(result_dir + "GC.pdf", bbox_inches = "tight")    

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_title("graph connectivity", fontsize = 20)
_ = ax.set_xticks([1,2,3])
_ = ax.set_xticklabels(["scMoMaT", "UINMF", "MultiMap"])
# _ = ax.set_xlabel("cluster", fontsize = 20)
_ = ax.set_ylabel("GC", fontsize = 20)
show_values_on_bars(ax)
fig.savefig(result_dir + "GC.png", bbox_inches = "tight")    

# NMI
nmi_scmomat = np.max(score.loc[score["methods"] == "scMoMaT", "NMI"].values)
nmi_uinmf = np.max(score.loc[score["methods"] == "UINMF", "NMI"].values)
nmi_multimap = np.max(score.loc[score["methods"] == "MultiMap", "NMI"].values)

fig = plt.figure(figsize = (4,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3], [nmi_scmomat, nmi_uinmf, nmi_multimap], width = 0.4)
barlist[0].set_color('r')    

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_ylim(0, 1)
ax.set_title("NMI", fontsize = 20)
_ = ax.set_xticks([1,2,3])
_ = ax.set_xticklabels(["scMoMaT", "UINMF", "MultiMap"])
# _ = ax.set_xlabel("cluster", fontsize = 20)
_ = ax.set_ylabel("NMI", fontsize = 20)
show_values_on_bars(ax)
fig.savefig(result_dir + "NMI.png", bbox_inches = "tight")    

# ARI
ari_scmomat = np.max(score.loc[score["methods"] == "scMoMaT", "ARI"].values)
ari_uinmf = np.max(score.loc[score["methods"] == "UINMF", "ARI"].values)
ari_multimap = np.max(score.loc[score["methods"] == "MultiMap", "ARI"].values)

fig = plt.figure(figsize = (4,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3], [ari_scmomat, ari_uinmf, ari_multimap], width = 0.4)
barlist[0].set_color('r')    

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_title("ARI", fontsize = 20)
_ = ax.set_xticks([1,2,3])
_ = ax.set_xticklabels(["scMoMaT", "UINMF", "MultiMap"])
ax.set_ylim(0, 1)
# _ = ax.set_xlabel("cluster", fontsize = 20)
_ = ax.set_ylabel("ARI", fontsize = 20)
show_values_on_bars(ax)
fig.savefig(result_dir + "ARI.png", bbox_inches = "tight")    

# LTA
lta_scmomat = np.max(scores.loc[scores["methods"] == "scMoMaT", "LTA"].values)
lta_uinmf = np.max(scores.loc[scores["methods"] == "UINMF", "LTA"].values)
lta_multimap = np.max(scores.loc[scores["methods"] == "MultiMap", "LTA"].values)

fig = plt.figure(figsize = (4,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3], [lta_scmomat, lta_uinmf, lta_multimap], width = 0.4)
barlist[0].set_color('r')    

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_title("Label Transfer Accuracy", fontsize = 20)
_ = ax.set_xticks([1,2,3])
_ = ax.set_xticklabels(["scMoMaT", "UINMF", "MultiMap"])
# _ = ax.set_xlabel("cluster", fontsize = 20)
_ = ax.set_ylabel("LTA", fontsize = 20)
show_values_on_bars(ax)
fig.savefig(result_dir + "LTA.png", bbox_inches = "tight")    


# %%
