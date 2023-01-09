# In[]
import sys, os
sys.path.append('../')
import torch
import numpy as np
from umap import UMAP
import time
import pandas as pd  
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import scmomat.model as model
import scmomat.utils as utils
import scmomat.bmk as bmk
import scmomat.umap_batch as umap_batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def lsi(counts, n_components = 30):
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD

    tfidf = TfidfTransformer(norm='l2', sublinear_tf=True)
    normed_count = tfidf.fit_transform(counts)

    # perform SVD on the sparse matrix
    lsi = TruncatedSVD(n_components=n_components + 1, random_state=42)
    lsi_r = lsi.fit_transform(normed_count)

    lsi.explained_variance_ratio_

    X_lsi = lsi_r[:, 1:]
    return X_lsi

# In[] 
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   NOTE: 1. Load dataset and running scmomat (without retraining, retraining see the third section)
#
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# NOTE: read in dataset
dir = '../data/real/diag/healthy_hema/topgenes_1000/BMMC/'
result_dir = "bmmc_healthyhema/scmomat/"
seurat_path = "bmmc_healthyhema/seurat/"
liger_path = "bmmc_healthyhema/liger/"
uinmf_path = "bmmc_healthyhema/uinmf_bin/" 
multimap_path = "bmmc_healthyhema/multimap/"
stabmap_path = "bmmc_healthyhema/stabmap/"

counts_rnas = []
counts_atacs = []
labels = []
n_batches = 2
for batch in range(1, n_batches+1):
    labels.append(pd.read_csv(os.path.join(dir, 'meta_c' + str(batch) + '.csv'), index_col=0)["BioClassification"].values.squeeze())
    
    try:
        counts_atac = sp.load_npz(os.path.join(dir, 'RxC' + str(batch) + ".npz")).toarray().T
        counts_atac = utils.preprocess(counts_atac, modality = "ATAC")   
    except:
        counts_atac = None
        
    try:
        counts_rna = sp.load_npz(os.path.join(dir, 'GxC' + str(batch) + ".npz")).toarray().T
        counts_rna = utils.preprocess(counts_rna, modality = "RNA", log = False)
    except:
        counts_rna = None
    
    counts_rnas.append(counts_rna)
    counts_atacs.append(counts_atac)

counts = {"rna":counts_rnas, "atac": counts_atacs}

A = sp.load_npz(os.path.join(dir, 'GxR.npz')).toarray()
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

# remove the index of label
for batch_id, label in enumerate(labels):
    labels[batch_id] = np.array([x[3:] for x in label])
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

# start_time = time.time()
# model1 = model.scmomat_model(counts = counts, K = K, batch_size = batchsize, interval = interval, lr = lr, lamb = lamb, seed = seed, device = device)
# losses1 = model1.train_func(T = T)
# end_time = time.time()
# print("running time: " + str(end_time - start_time))

# x = np.linspace(0, T, int(T/interval) + 1)
# plt.plot(x, losses1)

# # state dict does not include the scaling factor, etc
# torch.save(model1, result_dir + f'CFRM_{K}_{T}.pt')
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
# NOTE: Plot the result before post-processing
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

utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = result_dir + f'latent_separate_{K}_{T}.png', figsize = (10,15), axis_label = "UMAP", markerscale = 6)

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = result_dir + f'latent_batches_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = result_dir + f'latent_clusters_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6)

# In[] 
# NOTE: Post-processing, clustering, and plot the result after post-processing
plt.rcParams["font.size"] = 10

n_neighbors = 30
r = None

zs = []
for batch in range(n_batches):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)
s_pair_dist, knn_indices, knn_dists = utils.post_process(zs, n_neighbors, r = r, njobs = 8)


# load the score.csv that we calculated in advance to select the best resolution
# scores = pd.read_csv(result_dir + "score2.csv", index_col = 0)
# scores = scores[scores["methods"] == "scMoMaT"] 
# resolution = scores["resolution"].values[np.argmax(scores["NMI"].values.squeeze())]
# print(resolution)

# resolution = 1.0 for 1000 genes, 2.0 for 2000 genes
resolution = 0.95

labels_tmp = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
umap_op = umap_batch.UMAP(n_components = 2, n_neighbors = n_neighbors, min_dist = 0.2, random_state = 0, 
                metric='precomputed', knn_dists=knn_dists, knn_indices=knn_indices)
x_umap_scmomat = umap_op.fit_transform(s_pair_dist)


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
                      figsize = (15,15), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", alpha = 0.5)

utils.plot_latent_ext(x_umaps_scmomat, annos = labels, mode = "modality", save = result_dir + f'latent_batches_{K}_{T}_processed2.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", alpha = 0.5)

utils.plot_latent_ext(x_umaps_scmomat, annos = labels, mode = "joint", save = result_dir + f'latent_clusters_{K}_{T}_processed2.png', 
                      figsize = (13,7), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", alpha = 0.5)

utils.plot_latent_ext(x_umaps_scmomat, annos = leiden_labels, mode = "joint", save = result_dir + f'latent_leiden_clusters_{K}_{T}_{resolution}_processed2.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, alpha = 0.5)


# In[] 
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   NOTE: 2. Benchmarking with baseline methods
#
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# NOTE: Baseline methods
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
                      figsize = (15,15), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", colormap = "tab20b", alpha = 0.5)

utils.plot_latent_ext(uinmf_umaps, annos = labels, mode = "modality", save = uinmf_path + f'latent_batches_uinmf.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", colormap = "tab20b", alpha = 0.5)

utils.plot_latent_ext(uinmf_umaps, annos = labels, mode = "joint", save = uinmf_path + f'latent_clusters_uinmf.png', 
                      figsize = (13,7), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", colormap = "tab20b", alpha = 0.5)


# 2. Multimap
batches = pd.read_csv(multimap_path + "batch_id.csv", index_col = 0)
X_multimap = np.load(multimap_path + "multimap.npy")
G_multimap = sp.load_npz(multimap_path + "multimap_graph.npz").toarray()
X_multimaps = []
for batch in ["C1", "C2"]:
    X_multimaps.append(X_multimap[batches.values.squeeze() == batch, :])

utils.plot_latent_ext(X_multimaps, annos = labels, mode = "separate", save = multimap_path + f'latent_separate_multimap.png', 
                      figsize = (15,15), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", colormap = "tab20b", alpha = 0.5)

utils.plot_latent_ext(X_multimaps, annos = labels, mode = "modality", save = multimap_path + f'latent_batches_multimap.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", colormap = "tab20b", alpha = 0.5)

utils.plot_latent_ext(X_multimaps, annos = labels, mode = "joint", save = multimap_path + f'latent_clusters_multimap.png', 
                      figsize = (13,7), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", colormap = "tab20b", alpha = 0.5)


# 3. Seurat
seurat_pcas = [pd.read_csv(seurat_path + "seurat_pca_c1.txt", sep = "\t", index_col = 0).values, 
               pd.read_csv(seurat_path + "seurat_pca_c2.txt", sep = "\t", index_col = 0).values]
seurat_umaps = [pd.read_csv(seurat_path + "seurat_umap_c1.txt", sep = "\t", index_col = 0).values,
               pd.read_csv(seurat_path + "seurat_umap_c2.txt", sep = "\t", index_col = 0).values]


utils.plot_latent_ext(seurat_umaps, annos = labels, mode = "separate", save = seurat_path + f'latent_separate_seurat.png', 
                      figsize = (15,15), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", colormap = "tab20b", alpha = 0.5)

utils.plot_latent_ext(seurat_umaps, annos = labels, mode = "modality", save = seurat_path + f'latent_batches_seurat.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", colormap = "tab20b", alpha = 0.5)

utils.plot_latent_ext(seurat_umaps, annos = labels, mode = "joint", save = seurat_path + f'latent_clusters_seurat.png', 
                      figsize = (13,7), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", colormap = "tab20b", alpha = 0.5)


# 4. Liger
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
                      figsize = (15,15), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", colormap = "tab20b", alpha = 0.5)

utils.plot_latent_ext(liger_umaps, annos = labels, mode = "modality", save = liger_path + f'latent_batches_liger.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", colormap = "tab20b", alpha = 0.5)

utils.plot_latent_ext(liger_umaps, annos = labels, mode = "joint", save = liger_path + f'latent_clusters_liger.png', 
                      figsize = (13,7), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", colormap = "tab20b", alpha = 0.5)

# 5. Stabmap
stabmap_b1 = pd.read_csv(stabmap_path + "stab_b1.csv", index_col = 0).values
stabmap_b2 = pd.read_csv(stabmap_path + "stab_b2.csv", index_col = 0).values
stabmap_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(np.concatenate((stabmap_b1, stabmap_b2), axis = 0))
stabmap_umaps = []
for batch in range(0,2):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        stabmap_umaps.append(stabmap_umap[start_pointer:end_pointer,:])
    elif batch == 1:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        stabmap_umaps.append(stabmap_umap[start_pointer:,:])
    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        stabmap_umaps.append(stabmap_umap[start_pointer:end_pointer,:])

utils.plot_latent_ext(stabmap_umaps, annos = labels, mode = "separate", save = stabmap_path + f'latent_separate_stabmap.png', 
                      figsize = (15,15), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", colormap = "tab20b", alpha = 0.5)

utils.plot_latent_ext(stabmap_umaps, annos = labels, mode = "modality", save = stabmap_path + f'latent_batches_stabmap.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", colormap = "tab20b", alpha = 0.5)

utils.plot_latent_ext(stabmap_umaps, annos = labels, mode = "joint", save = stabmap_path + f'latent_clusters_stabmap.png', 
                      figsize = (13,7), axis_label = "UMAP", markerscale = 6, s = 3, label_inplace = True, text_size = "large", colormap = "tab20b", alpha = 0.5)



# In[] 
# n_neighbors affect the overall acc
n_neighbors = knn_indices.shape[1]
# NOTE: calculate benchmarking scores
# note that here the result change with the number of neighobrs, 30, 20, etc
# labels[1] = np.where(labels[1] == "Memory_CD8_T", "T_CD8_naive", labels[1])
# graph connectivity score (gc) measure the batch effect removal per cell identity
# 1. scMoMaT
# construct neighborhood graph from the post-processed latent space
knn_graph = np.zeros((knn_indices.shape[0], knn_indices.shape[0]))
knn_graph[np.arange(knn_indices.shape[0])[:, None], knn_indices] = 1
gc_scmomat = bmk.graph_connectivity(G = knn_graph, groups = np.concatenate(labels, axis = 0))
print('GC (scmomat): {:.3f}'.format(gc_scmomat))

# 2. Seurat
gc_seurat = bmk.graph_connectivity(X = np.concatenate(seurat_pcas, axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (Seurat): {:.3f}'.format(gc_seurat))

# 3. Liger
gc_liger = bmk.graph_connectivity(X = np.concatenate((H1_liger, H2_liger), axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (Liger): {:.3f}'.format(gc_liger))

# 4. UINMF
gc_uinmf = bmk.graph_connectivity(X = np.concatenate((H1_uinmf, H2_uinmf), axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (UINMF): {:.3f}'.format(gc_uinmf))

# 5. Multimap
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

# 6. Stabmap
gc_stabmap = bmk.graph_connectivity(X = np.concatenate((stabmap_b1, stabmap_b2), axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (Stabmap): {:.3f}'.format(gc_stabmap))

# Batch effect removal regardless of cell identity
# Graph iLISI

# Conservation of biological identity
# NMI and ARI
# 1. scMoMaT
nmi_scmomat = []
ari_scmomat = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_scmomat = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
    nmi_scmomat.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_scmomat))
    ari_scmomat.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_scmomat))
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

# 6. Stabmap
nmi_stabmap = []
ari_stabmap = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_stabmap = utils.leiden_cluster(X = np.concatenate((stabmap_b1, stabmap_b2), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_stabmap.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_stabmap))
    ari_stabmap.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_stabmap))
print('NMI (Stabmap): {:.3f}'.format(max(nmi_stabmap)))
print('ARI (Stabmap): {:.3f}'.format(max(ari_stabmap)))


# Label transfer accuracy
# randomly select a half of cells as query
np.random.seed(0)
query_cell = np.array([False] * knn_indices.shape[0])
query_cell[np.random.choice(np.arange(knn_indices.shape[0]), size = int(0.5 * knn_indices.shape[0]), replace = False)] = True
training_cell = (1 - query_cell).astype(np.bool)
query_label = np.concatenate(labels)[query_cell]
training_label = np.concatenate(labels)[training_cell]

# NOTE: KNN graph should be constructed between train and query cells. We should have n_neighbors train cells around each query cell, and then vote
# however, the pre-reconstructed knn graph for scMoMaT and MultiMap find n_neighbors from all cells (train+query), it's hard to modify pre-reconstructed graph to match the requirement.
# We use the pre-reconstructed graph directly and ignore the query cells when voting, to methods still have the same number of n_neighbors
# scmomat
knn_graph = np.zeros((knn_indices.shape[0], knn_indices.shape[0]))
knn_graph[np.arange(knn_indices.shape[0])[:, None], knn_indices] = 1
knn_graph = knn_graph[query_cell, :][:, training_cell]
lta_scmomat = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, knn_graph = knn_graph)

# Seurat
lta_seurat = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, 
                                  z_query = np.concatenate(seurat_pcas, axis = 0)[query_cell,:],
                                  z_train = np.concatenate(seurat_pcas, axis = 0)[training_cell,:])

# Liger
lta_liger = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, 
                                  z_query = np.concatenate((H1_liger, H2_liger), axis = 0)[query_cell,:],
                                  z_train = np.concatenate((H1_liger, H2_liger), axis = 0)[training_cell,:])

# UINMF
lta_uinmf = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, 
                                  z_query = np.concatenate((H1_uinmf, H2_uinmf), axis = 0)[query_cell,:],
                                  z_train = np.concatenate((H1_uinmf, H2_uinmf), axis = 0)[training_cell,:])

# MultiMap
G_multimap = sp.load_npz(multimap_path + "multimap_graph.npz").toarray()
knn_indices_multimap = G_multimap.argsort(axis = 1)[:, -n_neighbors:]
knn_graph_multimap = np.zeros_like(G_multimap)
knn_graph_multimap[np.arange(knn_indices_multimap.shape[0])[:, None], knn_indices_multimap] = 1
lta_multimap = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, knn_graph = knn_graph_multimap[query_cell, :][:, training_cell])
lta_multimap2 = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, 
                                  z_query = np.concatenate(X_multimaps, axis = 0)[query_cell,:],
                                  z_train = np.concatenate(X_multimaps, axis = 0)[training_cell,:])

# UINMF
lta_stabmap = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, 
                                  z_query = np.concatenate((stabmap_b1, stabmap_b2), axis = 0)[query_cell,:],
                                  z_train = np.concatenate((stabmap_b1, stabmap_b2), axis = 0)[training_cell,:])

print("Label transfer accuracy (scMoMaT): {:.3f}".format(lta_scmomat))
print("Label transfer accuracy (Seurat): {:.3f}".format(lta_seurat))
print("Label transfer accuracy (Liger): {:.3f}".format(lta_liger))
print("Label transfer accuracy (UINMF): {:.3f}".format(lta_uinmf))
print("Label transfer accuracy (MultiMap Graph): {:.3f}".format(lta_multimap))
print("Label transfer accuracy (MultiMap): {:.3f}".format(lta_multimap2))
print("Label transfer accuracy (Stabmap): {:.3f}".format(lta_stabmap))


scores = pd.DataFrame(columns = ["methods", "resolution", "NMI", "ARI", "GC"])
scores["NMI"] = np.array(nmi_scmomat + nmi_seurat + nmi_liger + nmi_uinmf + nmi_multimap + nmi_stabmap)
scores["ARI"] = np.array(ari_scmomat + ari_seurat + ari_liger + ari_uinmf + ari_multimap + ari_stabmap)
scores["GC"] = np.array([gc_scmomat] * len(nmi_scmomat) + [gc_seurat] * len(nmi_seurat) + [gc_liger] * len(nmi_liger) + [gc_uinmf] * len(nmi_uinmf) + [gc_multimap] * len(nmi_multimap) + [gc_stabmap] * len(nmi_stabmap))
scores["LTA"] = np.array([lta_scmomat] * len(nmi_scmomat) + [lta_seurat] * len(nmi_seurat) + [lta_liger] * len(nmi_liger) + [lta_uinmf] * len(nmi_uinmf) + [lta_multimap] * len(ari_multimap) + [lta_stabmap] * len(nmi_stabmap))
scores["resolution"] = np.array([x for x in np.arange(0.1, 10, 0.5)] * 6)
scores["methods"] = np.array(["scMoMaT"] * len(nmi_scmomat) + ["Seurat"] * len(nmi_seurat) + ["Liger"] * len(nmi_liger) + ["UINMF"] * len(nmi_uinmf) + ["MultiMap"] * len(nmi_multimap) + ["Stabmap"] * len(nmi_stabmap))
scores.to_csv(result_dir + "score.csv")

# In[]
# GC
plt.rcParams["font.size"] = 15
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.4f}'.format(p.get_height())
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
gc_liger = np.max(score.loc[score["methods"] == "Liger", "GC"].values)
gc_stabmap = np.max(score.loc[score["methods"] == "Stabmap", "GC"].values)

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3,4], [gc_scmomat, gc_uinmf, gc_multimap, gc_liger, gc_stabmap], width = 0.4)
barlist[0].set_color('r')
fig.savefig(result_dir + "GC.pdf", bbox_inches = "tight")    

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_title("graph connectivity", fontsize = 20)
_ = ax.set_xticks([1,2,3,4,5])
_ = ax.set_xticklabels(["scMoMaT", "UINMF", "MultiMap", "Liger", "Stabmap"])
# _ = ax.set_xlabel("cluster", fontsize = 20)
_ = ax.set_ylabel("GC", fontsize = 20)
show_values_on_bars(ax)
fig.savefig(result_dir + "GC.png", bbox_inches = "tight")    

# NMI
nmi_scmomat = np.max(score.loc[score["methods"] == "scMoMaT", "NMI"].values)
nmi_uinmf = np.max(score.loc[score["methods"] == "UINMF", "NMI"].values)
nmi_multimap = np.max(score.loc[score["methods"] == "MultiMap", "NMI"].values)
nmi_liger = np.max(score.loc[score["methods"] == "Liger", "NMI"].values)
nmi_stabmap = np.max(score.loc[score["methods"] == "Stabmap", "NMI"].values)

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3,4,5], [nmi_scmomat, nmi_uinmf, nmi_multimap, nmi_liger, nmi_stabmap], width = 0.4)
barlist[0].set_color('r')    

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_title("NMI", fontsize = 20)
_ = ax.set_xticks([1,2,3,4])
_ = ax.set_xticklabels(["scMoMaT", "UINMF", "MultiMap", "Liger"])
# _ = ax.set_xlabel("cluster", fontsize = 20)
_ = ax.set_ylabel("NMI", fontsize = 20)
show_values_on_bars(ax)
fig.savefig(result_dir + "NMI.png", bbox_inches = "tight")    

# ARI
ari_scmomat = np.max(score.loc[score["methods"] == "scMoMaT", "ARI"].values)
ari_uinmf = np.max(score.loc[score["methods"] == "UINMF", "ARI"].values)
ari_multimap = np.max(score.loc[score["methods"] == "MultiMap", "ARI"].values)
ari_liger = np.max(score.loc[score["methods"] == "Liger", "ARI"].values)

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3,4], [ari_scmomat, ari_uinmf, ari_multimap, ari_liger], width = 0.4)
barlist[0].set_color('r')    

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_title("ARI", fontsize = 20)
_ = ax.set_xticks([1,2,3,4])
_ = ax.set_xticklabels(["scMoMaT", "UINMF", "MultiMap", "Liger"])
# _ = ax.set_xlabel("cluster", fontsize = 20)
_ = ax.set_ylabel("ARI", fontsize = 20)
show_values_on_bars(ax)
fig.savefig(result_dir + "ARI.png", bbox_inches = "tight")    

# LTA
lta_scmomat = np.max(score.loc[score["methods"] == "scMoMaT", "LTA"].values)
lta_uinmf = np.max(score.loc[score["methods"] == "UINMF", "LTA"].values)
lta_multimap = np.max(score.loc[score["methods"] == "MultiMap", "LTA"].values)
lta_liger = np.max(score.loc[score["methods"] == "Liger", "LTA"].values)

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3,4], [lta_scmomat, lta_uinmf, lta_multimap, lta_liger], width = 0.4)
barlist[0].set_color('r')    

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_title("LTA", fontsize = 20)
_ = ax.set_xticks([1,2,3,4])
_ = ax.set_xticklabels(["scMoMaT", "UINMF", "MultiMap", "Liger"])
# _ = ax.set_xlabel("cluster", fontsize = 20)
_ = ax.set_ylabel("LTA", fontsize = 20)
show_values_on_bars(ax)
fig.savefig(result_dir + "LTA.png", bbox_inches = "tight")    


# In[] 
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   NOTE: 3. Retraining scmomat 
#
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# read in dataset
counts_rnas = []
counts_atacs = []
counts_motifs = []
labels = []
n_batches = 2
ks = [5, 10, 20, 30, 40, 50]
knnp_orig = []
knnp_leiden = []
silhouette_orig = []
silhouette_leiden = []
for batch in range(n_batches):
    labels.append(pd.read_csv(os.path.join(dir, 'meta_c' + str(batch + 1) + '.csv'), index_col=0)["BioClassification"].values.squeeze())
    
    try:
        counts_atac = sp.load_npz(os.path.join(dir, 'RxC' + str(batch + 1) + ".npz")).toarray().T
        counts_atac = utils.preprocess(counts_atac, modality = "ATAC")
        # Plot
        x_lsi = lsi(counts_atac, n_components = 30)
        x_umap = UMAP(n_components = 2, min_dist = 0.1, random_state = 0).fit_transform(x_lsi)
        print("umap atac for batch" + str(batch + 1))
        utils.plot_latent_ext([x_umap], annos = [labels[batch]], mode = "joint", save = result_dir + f'RxC{batch+1}.png', figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 10, label_inplace = False)
        utils.plot_latent_ext([x_umap], annos = [leiden_labels[batch]], mode = "joint", save = result_dir + f'RxC{batch+1}_leiden.png', figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 10, label_inplace = False)

        silhouette_orig.append(bmk.silhouette(x_lsi, labels[batch]))
        silhouette_leiden.append(bmk.silhouette(x_lsi, leiden_labels[batch]))        
        print("silhouette score (Original cluster): {:.3f}".format(silhouette_orig[-1]))
        print("silhouette score (Leiden cluster): {:.3f}".format(silhouette_leiden[-1]))
        knnp_orig.append([])
        knnp_leiden.append([])
        for k in ks:
            knnp_orig[-1].append(bmk.knn_purity(X = x_lsi, label = labels[batch], k = k))
            knnp_leiden[-1].append(bmk.knn_purity(X = x_lsi, label = leiden_labels[batch], k = k))
            print(f"k = {k}")
            print("knn purity score (Original cluster): {:.3f}".format(knnp_orig[-1][-1]))
            print("knn purity score (Leiden cluster): {:.3f}".format(knnp_leiden[-1][-1]))
   
    except:
        counts_atac = None
        
    try:
        counts_rna = sp.load_npz(os.path.join(dir, 'GxC' + str(batch + 1) + ".npz")).toarray().T
        counts_rna = utils.preprocess(counts_rna, modality = "RNA", log = False)
        # Plot
        x_pca = PCA(n_components = 30).fit_transform(np.log1p(counts_rna))
        x_umap = UMAP(n_components = 2, min_dist = 0.1, random_state = 0).fit_transform(x_pca)
        print("umap rna for batch" + str(batch + 1))
        utils.plot_latent_ext([x_umap], annos = [labels[batch]], mode = "joint", save = result_dir + f'GxC{batch+1}', figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 10, label_inplace = False)
        utils.plot_latent_ext([x_umap], annos = [leiden_labels[batch]], mode = "joint", save = result_dir + f'GxC{batch+1}_leiden.png', figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 10, label_inplace = False)

        silhouette_orig.append(bmk.silhouette(x_pca, labels[batch]))
        silhouette_leiden.append(bmk.silhouette(x_pca, leiden_labels[batch]))
        print("silhouette score (Original cluster): {:.3f}".format(silhouette_orig[-1]))
        print("silhouette score (Leiden cluster): {:.3f}".format(silhouette_leiden[-1]))
        knnp_orig.append([])
        knnp_leiden.append([])
        for k in ks:
            knnp_orig[-1].append(bmk.knn_purity(X = x_pca, label = labels[batch], k = k))
            knnp_leiden[-1].append(bmk.knn_purity(X = x_pca, label = leiden_labels[batch], k = k))
            print(f"k = {k}")
            print("knn purity score (Original cluster): {:.3f}".format(knnp_orig[-1][-1]))
            print("knn purity score (Leiden cluster): {:.3f}".format(knnp_leiden[-1][-1]))

    except:
        counts_rna = None
    
    try:
        counts_motif = pd.read_csv(dir + r'C{}xM.csv'.format(batch + 1), index_col = 0)
        # there might be small amount of na
        counts_motif = counts_motif.fillna(0)
        motifs = counts_motif.columns.values
        counts_motif = counts_motif.values
        # chromVAR provide the z-score, which has negative values
        counts_motif = (counts_motif - np.min(counts_motif))/(np.max(counts_motif) - np.min(counts_motif) + 1e-6)
    except:
        counts_motif = None
    
    counts_rnas.append(counts_rna)
    counts_atacs.append(counts_atac)
    counts_motifs.append(counts_motif)

counts = {"rna":counts_rnas, "atac": counts_atacs, "motif": counts_motifs}

A = sp.load_npz(os.path.join(dir, 'GxR.npz')).toarray()
interacts = None

genes = pd.read_csv(dir + "genes.txt", header = None).values.squeeze()
regions = pd.read_csv(dir + "regions.txt", header = None).values.squeeze()

feats_name = {"rna": genes, "atac": regions, "motif": motifs}
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

knnp_orig = np.array(knnp_orig)
knnp_leiden = np.array(knnp_leiden)
knnp_orig = np.mean(knnp_orig, axis = 0)
knnp_leiden = np.mean(knnp_leiden, axis = 0)

from matplotlib.ticker import FormatStrFormatter
plt.rcParams["font.size"] = 15
fig = plt.figure(figsize = (7, 5))
ax = fig.add_subplot()
ax.plot(ks, knnp_orig, label = "Original BMMC")
ax.plot(ks, knnp_leiden, label = "scMoMaT")
ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
ax.set_title("KNN purity")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.savefig(result_dir + "knn_purity.png", bbox_inches = "tight")

print("silhouette score (Original cluster): {:.3f}".format(np.mean(np.array(silhouette_orig))))
print("silhouette score (Leiden cluster): {:.3f}".format(np.mean(np.array(silhouette_leiden))))

# In[] retrain model, you can incorporate new matrices 
lamb = 0.01

# the leiden label is the one produced by the best resolution
model2 = model.scmomat_retrain(model = model1, counts =  counts, labels = leiden_labels, lamb = lamb, device = device)
losses = model2.train(T = 4000)

x = np.linspace(0, 4000, int(4000/interval) + 1)
plt.plot(x, losses)

C_feats = {}
for mod in model2.mods:
    C_feat = model2.softmax(model2.C_feats[mod]).data.cpu().numpy() @ model2.A_assos["shared"].data.cpu().numpy().T 
    C_feats[mod] = pd.DataFrame(data = C_feat, index = model2.feats_name[mod], columns = ["cluster_" + str(i) for i in range(C_feat.shape[1])])

# In[]
C_gene = C_feats["rna"]
utils.plot_feat_score(C_gene, n_feats = 20, figsize= (15,30), save_as = result_dir + "C_gene.pdf", title = None)

C_motif = C_feats["motif"]
utils.plot_feat_score(C_motif, n_feats = 20, figsize= (20,30), save_as = result_dir + "C_motif.pdf", title = None)

C_region = C_feats["atac"]

# C_gene.to_csv(result_dir + "C_gene.csv")
# C_motif.to_csv(result_dir + "C_motif.csv")
# C_region.to_csv(result_dir + "C_region.csv")

C_gene = pd.read_csv(result_dir + "C_gene.csv", index_col = 0)
C_motif = pd.read_csv(result_dir + "C_motif.csv", index_col = 0)
C_region = pd.read_csv(result_dir + "C_region.csv", index_col = 0)

# TODO: normalize between 0 and 1
C_gene.values[:] = C_gene.values/np.sum(C_gene.values, axis = 0, keepdims = True)
C_motif.values[:] = C_motif.values/np.sum(C_motif.values, axis = 0, keepdims = True)
C_region.values[:] = C_region.values/np.sum(C_region.values, axis = 0, keepdims = True)


# In[] 
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   NOTE: 4. Analyze retraining results 
#
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Factors Differentiation trajectory, see figure in [Wiki](https://en.wikipedia.org/wiki/Hematopoietic_stem_cell)
# B cells and pre-B cells: CD19 (not included), CD79A (included), CD37 (not included) Pax5 (included)

# CMP/LMPP CMP (common myeloid progenitor) LMPP (lymphoid primed multipotent progenitor/Lymphoid multipotent progenitors)

# CLP (common lymphoid progenitor cell)

# B (lymphocytes) cell lineage (HSC -> MPP -> LMPP -> ELP -> CLP -> B-biased -> Pro-B -> Pre-B -> Immature B -> Mature B, (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2854230/))

# T/NK cell lineage (HSC -> MPP -> LMPP -> CLP -> T cells/NK) [Wiki](https://en.wikipedia.org/wiki/Hematopoietic_stem_cell)

# Erythroid lineage (HSC -> MPP -> CMP -> Erythroid): 

# GMP: (granulocyte-monocyte progenitors), mature into maturing into a variety of white blood cells: Neutrophile, Basophile, macrophages are white blood cells

plt.rcParams["font.size"] = 20

# factor 8: NK: GNLY (included), NKG7 (included), KLRB1 (included), KLRD1 (included), KLRF1 (included)
for gene in ['GNLY', 'KLRD1', 'TBX21']: # 'CD37' not included
    expr = [counts["rna"][0][:, counts["feats_name"]["rna"] == gene].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == gene].squeeze() ]
    utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = expr, mode = "joint", save = result_dir + gene + ".png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "Reds", title = gene)
    fig = utils.plot_factor(C_gene, markers = [gene], cluster = 8, figsize = (7,5))
    fig.savefig(result_dir + gene + "_score.png", bbox_inches = "tight")

# factor 0, 3, 6: T cells
for gene in ["CD3D", "CD3E", "CD28"]:
    expr = [counts["rna"][0][:, counts["feats_name"]["rna"] == gene].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == gene].squeeze() ]
    utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = expr, mode = "joint", save = result_dir + gene + ".png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "Reds", title = gene)
    fig = utils.plot_factor(C_gene, markers = [gene], cluster = [0,3,6], figsize = (7,5))
    fig.savefig(result_dir + gene + "_score.png", bbox_inches = "tight")    

# factor 6: CD8+ T cells
CD8A = [counts["rna"][0][:, counts["feats_name"]["rna"] == "CD8A"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "CD8A"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CD8A, mode = "joint", save = result_dir + "CD8A.png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "Reds", title = "CD8A")
fig = utils.plot_factor(C_gene, markers = ["CD8A"], cluster = 6, figsize = (7,5))
fig.savefig(result_dir + "CD8A_score.png", bbox_inches = "tight")

CD8B = [counts["rna"][0][:, counts["feats_name"]["rna"] == "CD8B"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "CD8B"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CD8B, mode = "joint", save = result_dir + "CD8B.png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "Reds", title = "CD8B")
fig = utils.plot_factor(C_gene, markers = ["CD8B"], cluster = 6, figsize = (7,5))
fig.savefig(result_dir + "CD8B_score.png", bbox_inches = "tight")


# factor 0, 3: CD4+ Naive T cells
for gene in ["CCR7", "TCF7"]:
    expr = [counts["rna"][0][:, counts["feats_name"]["rna"] == gene].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == gene].squeeze() ]
    utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = expr, mode = "joint", save = result_dir + gene + ".png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "Reds", title = gene)
    fig = utils.plot_factor(C_gene, markers = [gene], cluster = [0,3], figsize = (7,5))
    fig.savefig(result_dir + gene + "_score.png", bbox_inches = "tight")    

# factor 7: B cells, factor 6: pre-B cells
# A guid to B cell marker: https://www.biocompare.com/Editorial-Articles/570578-A-Guide-to-B-Cell-Markers/
# find the differentially expressed genes between pre-B and B
# Collectively, the developmental stages give rise to populations of Pro B, Pre B, and Immature B cells. 
# Markers reported for these stages include CD22, IL7R, CD34, CD38, CD79, and MME. 
# B cells that complete this program make their way toward the spleen in transitional stages (T1, T2, T3) and express CD19+, IgDlo/+, CD27–, CD24++, CD38++.

# factor 7: mature B cell CD79A CD37 
# the gene of IgM (i.e. Ighm) is not included in the original gene set
# CD37 should be included
for gene in ['CD79A', 'PAX5']: # 'CD37' not included
    expr = [counts["rna"][0][:, counts["feats_name"]["rna"] == gene].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == gene].squeeze() ]
    utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = expr, mode = "joint", save = result_dir + gene + ".png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "Reds", title = gene)
    fig = utils.plot_factor(C_gene, markers = [gene], cluster = [5, 7], figsize = (7,5))
    fig.savefig(result_dir + gene + "_score.png", bbox_inches = "tight")

# pre-B cells 
# print(genes[np.argsort(np.abs(C_gene.values[:, 7] - C_gene.values[:, 6]).squeeze())[::-1]])
# de_gene = genes[np.argsort(np.abs(C_gene.values[:, 7] - C_gene.values[:, 6]).squeeze())[::-1]]
for gene in ['MME', 'CD38', 'CD34']: # IL7R, CD22, CD79B
    expr = [counts["rna"][0][:, counts["feats_name"]["rna"] == gene].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == gene].squeeze() ]
    utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = expr, mode = "joint", save = result_dir + gene + ".png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "Reds", title = gene)
    fig = utils.plot_factor(C_gene, markers = [gene], cluster = [5, 7], figsize = (7,5))
    fig.savefig(result_dir + gene + "_score.png", bbox_inches = "tight")

# factor 1 & 2 & 10 Monocyte
# CCL2,CD14+ monocytes, not included
# S100A8,CD14+ monocytes
# S100A9,CD14+ monocytes,
# CD14,CD14+ monocytes,
# LYZ,CD14+ monocytes,
# LGALS3,CD14+ monocytes,
# S100A12,CD14+ monocytes,
# VMO1,CD16+ monocytes, not included
# FCGR3A,FCGR3A+ monocytes,
# MS4A7,FCGR3A+ monocytes, not included
# CEBPB not included motif
for gene in ['S100A8', 'S100A9', 'CD14', 'LYZ', 'LGALS3', 'S100A12', 'FCGR3A']: 
    expr = [counts["rna"][0][:, counts["feats_name"]["rna"] == gene].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == gene].squeeze() ]
    utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = expr, mode = "joint", save = result_dir + gene + ".png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "Reds", title = gene)
    fig = utils.plot_factor(C_gene, markers = [gene], cluster = [1, 2, 10], figsize = (7,5))
    fig.savefig(result_dir + gene + "_score.png", bbox_inches = "tight")

# factor 9, pDC, ID3, CLEC4C (BDCA-2), PTPRC, ptprs (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4547994/)
for gene in ['PTPRS']: # ['ID3', 'STAT1']: 
    expr = [counts["rna"][0][:, counts["feats_name"]["rna"] == gene].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == gene].squeeze() ]
    utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = expr, mode = "joint", save = result_dir + gene + ".png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "Reds", title = gene)
    fig = utils.plot_factor(C_gene, markers = [gene], cluster = 9, figsize = (7,5))
    fig.savefig(result_dir + gene + "_score.png", bbox_inches = "tight")

# factor 4, erythroid/HSC lineage: GATA1 (not included)
for gene in ['HOXB2']:
    expr = [counts["rna"][0][:, counts["feats_name"]["rna"] == gene].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == gene].squeeze() ]
    utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = expr, mode = "joint", save = result_dir + gene + ".png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "Reds", title = gene)
    fig = utils.plot_factor(C_gene, markers = [gene], cluster = 4, figsize = (7,5))
    fig.savefig(result_dir + gene + "_score.png", bbox_inches = "tight")

# In[] Motif

# factor 4 erythroid/HSC lineage: MA0140.2_GATA1::TAL1 (erythroid lineage), MA0907.1_HOXC13 (HOX family)

# factor 1 CMP/LMPP/Monocyte lineage: Myeloid (MA0466.2_CEBPB),
for motif in ["MA0466.2_CEBPB"]:
    expr = counts["motif"][0][:, counts["feats_name"]["motif"] == motif].squeeze()
    utils.plot_latent_continuous([x_umaps_scmomat[:2][0]], annos = [expr], mode = "joint", save = result_dir + motif + ".png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "Reds", title = motif)
    fig = utils.plot_factor(C_motif, markers = [motif], cluster = [1,2,10], figsize = (7,5))
    fig.savefig(result_dir + motif + "_score.png", bbox_inches = "tight")
 

# factor 0, 3, 6: T cell motif
for motif in ["MA0850.1_FOXP3", "MA0523.1_TCF7L2", "MA0033.2_FOXL1"]:
    expr = counts["motif"][0][:, counts["feats_name"]["motif"] == motif].squeeze()
    utils.plot_latent_continuous([x_umaps_scmomat[:2][0]], annos = [expr], mode = "joint", save = result_dir + motif + ".png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "Reds", title = motif)
    fig = utils.plot_factor(C_motif, markers = [motif], cluster = [0,3,6], figsize = (7,5))
    fig.savefig(result_dir + motif + "_score.png", bbox_inches = "tight")

# factor 8: NK motif MA0800.1_EOMES
for motif in ["MA0800.1_EOMES"]:
    expr = counts["motif"][0][:, counts["feats_name"]["motif"] == motif].squeeze()
    utils.plot_latent_continuous([x_umaps_scmomat[:2][0]], annos = [expr], mode = "joint", save = result_dir + motif + ".png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "Reds", title = motif)
    fig = utils.plot_factor(C_motif, markers = [motif], cluster = 8, figsize = (7,5))
    fig.savefig(result_dir + motif + "_score.png", bbox_inches = "tight")
# %%
