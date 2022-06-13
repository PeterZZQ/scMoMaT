# In[]
import sys, os
sys.path.append('../')
sys.path.append('../src/')

import numpy as np
from sklearn.decomposition import PCA
import umap_batch
from umap import UMAP
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd  
import scipy.sparse as sp
import bmk
from scipy.io import mmwrite, mmread

import model
import utils
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.size"] = 10
import warnings
warnings.filterwarnings("ignore")

def lsi(counts):
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD

    tfidf = TfidfTransformer(norm='l2', sublinear_tf=True)
    normed_count = tfidf.fit_transform(counts)

    # perform SVD on the sparse matrix
    lsi = TruncatedSVD(n_components=50, random_state=42)
    lsi_r = lsi.fit_transform(normed_count)

    lsi.explained_variance_ratio_

    X_lsi = lsi_r[:, 1:]
    return X_lsi

# In[]
dir = "../data/simulated/6b16c_test_9_large/unequal2/"
result_dir = "simulated/6b16c_9_large2_2"
scmomat_dir = result_dir + "/scmomat/"

if not os.path.exists(scmomat_dir):
    os.makedirs(scmomat_dir)

n_batches = 6
counts_rnas = []
counts_atacs = []
labels = []
for batch in range(n_batches):        
    label = pd.read_csv(os.path.join(dir, 'cell_label' + str(batch + 1) + '.txt'), index_col=0, sep = "\t")["pop"].values.squeeze()
    labels.append(label)

    try:
        counts_atac = np.loadtxt(os.path.join(dir, 'RxC' + str(batch + 1) + ".txt"), delimiter = "\t").T
        counts_atac = utils.preprocess(counts_atac, modality = "ATAC")
        print("read atac for batch" + str(batch + 1))
        # x_lsi = lsi(counts_atac)
        # x_umap = UMAP(n_components = 2, min_dist = 0.1, random_state = 0).fit_transform(x_lsi)
        # print("umap atac for batch" + str(batch + 1))
        # utils.plot_latent_ext([x_umap], annos = [labels[-1]], mode = "joint", save = scmomat_dir + f'RxC{batch+1}', figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True)
    except:
        counts_atac = None
        
    try:
        counts_rna = np.loadtxt(os.path.join(dir, 'GxC' + str(batch + 1) + ".txt"), delimiter = "\t").T
        print("read rna for batch" + str(batch + 1))
        counts_rna = utils.preprocess(counts_rna, modality = "RNA", log = False)
        # x_pca = PCA(n_components = 30).fit_transform(np.log1p(counts_rna))
        # x_umap = UMAP(n_components = 2, min_dist = 0.1, random_state = 0).fit_transform(x_pca)
        # print("umap rna for batch" + str(batch + 1))
        # utils.plot_latent_ext([x_umap], annos = [labels[-1]], mode = "joint", save = scmomat_dir + f'GxC{batch+1}', figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True)

    except:
        counts_rna = None
    
    # preprocess the count matrix
    counts_rnas.append(counts_rna)
    counts_atacs.append(counts_atac)

counts = {"rna":counts_rnas, "atac": counts_atacs}

# # diagonal integration
# counts["rna"][0] = None
# counts["rna"][1] = None
# counts["rna"][2] = None
# counts["atac"][3] = None 
# counts["atac"][4] = None
# counts["atac"][5] = None

# diagonal with partial shared
counts["rna"][0] = None
counts["rna"][1] = None
counts["rna"][2] = None
counts["atac"][4] = None
counts["atac"][5] = None

# No need for pseudo-count matrix
A = np.loadtxt(os.path.join(dir, 'region2gene.txt'), delimiter = "\t").T

# CALCULATE PSEUDO-SCRNA-SEQ
for idx in range(len(counts["atac"])):
    if (counts["rna"][idx] is None) & (counts["atac"][idx] is not None):
        counts["rna"][idx] = counts["atac"][idx] @ A.T
        #BINARIZE, still is able to see the cluster pattern, much denser than scRNA-Seq (cluster pattern clearer)
        counts["rna"][idx] = (counts["rna"][idx]!=0).astype(int)

# obtain the feature name
genes = np.array(["gene_" + str(x) for x in range(counts["rna"][-1].shape[1])])
regions = np.array(["region_" + str(x) for x in range(counts["atac"][0].shape[1])])

feats_name = {"rna": genes, "atac": regions}
counts["feats_name"] = feats_name

counts["nbatches"] = n_batches

# only ATAC
# counts = {"atac": counts["atac"], "feats_name": {"atac": feats_name["atac"]}, "nbatches": n_batches}
# only RNA
# counts = {"rna": counts["rna"], "feats_name": {"rna": feats_name["rna"]}, "nbatches": n_batches}

# # In[] Check batch effect between scATAC-seq batches
# umap_op = UMAP(n_components = 2, n_neighbors = 30, min_dist = 0.2, random_state = 0) 
# x_umap = umap_op.fit_transform(np.concatenate(counts["atac"], axis = 0))

# # separate into batches
# x_umaps = []
# leiden_labels = []
# for batch in range(n_batches):
#     if batch == 0:
#         start_pointer = 0
#         end_pointer = start_pointer + counts["atac"][batch].shape[0]
#         x_umaps.append(x_umap[start_pointer:end_pointer,:])
        
#     elif batch == (n_batches - 1):
#         start_pointer = start_pointer + counts["atac"][batch - 1].shape[0]
#         x_umaps.append(x_umap[start_pointer:,:])
        
#     else:
#         start_pointer = start_pointer + counts["atac"][batch - 1].shape[0]
#         end_pointer = start_pointer + counts["atac"][batch].shape[0]
#         x_umaps.append(x_umap[start_pointer:end_pointer,:])

# utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = None, figsize = (15,10), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True)
# utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = None, figsize = (15,10), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True)


# In[]
alpha = [1000, 1, 5]
batchsize = 0.1
run = 0
K = 30
K = 20
interval = 1000
T = 4000
lr = 1e-2

start_time = time.time()
model1 = model.cfrm_vanilla(counts = counts, K = K, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run, device = device).to(device)
losses1 = model1.train_func(T = T)
end_time = time.time()
print("running time: " + str(end_time - start_time))

x = np.linspace(0, T, int(T/interval)+1)
plt.plot(x, losses1)

torch.save(model1, scmomat_dir + f'CFRM_{K}_{T}.pt')
model1 = torch.load(scmomat_dir + f'CFRM_{K}_{T}.pt')

# # In[] Sanity check, the scales should be positive, A_assos should also be positive
# for mod in model1.A_assos.keys():
#     if mod != "shared":
#         print(torch.min(model1.A_assos["shared"] + model1.A_assos[mod]).item())

# for mod in model1.A_assos.keys():
#     if mod != "shared":
#         print(torch.mean(model1.A_assos["shared"] + model1.A_assos[mod]).item())

# for mod in model1.A_assos.keys():
#     if mod != "shared":
#         print(torch.max(model1.A_assos["shared"] + model1.A_assos[mod]).item())

# print(model1.scales)

# In[]
plt.rcParams["font.size"] = 10
umap_op = UMAP(n_components = 2, n_neighbors = 30, min_dist = 0.2, random_state = 0) 
zs = []
for batch in range(n_batches):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)
    
x_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

# separate into batches
x_umaps = []
leiden_labels = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
        
    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
        
    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])

utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = scmomat_dir + f'latent_separate_{K}_{T}.png', figsize = (15,30), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True)
utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = scmomat_dir + f'latent_batches_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True)

# In[]
import importlib 
importlib.reload(utils)
n_neighbors = 30
r = 0.7

zs = []
for batch in range(n_batches):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)

s_pair_dist, knn_indices, knn_dists = utils.post_process(zs, n_neighbors, njobs = 8, r = r)

resolution = 0.5
labels_tmp = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
umap_op = umap_batch.UMAP(n_components = 2, n_neighbors = n_neighbors, min_dist = 0.30, random_state = 0, 
                metric='precomputed', knn_dists=knn_dists, knn_indices=knn_indices)
x_umap = umap_op.fit_transform(s_pair_dist)


# separate into batches
x_umaps = []
leiden_labels = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
        leiden_labels.append(labels_tmp[start_pointer:end_pointer])

    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
        leiden_labels.append(labels_tmp[start_pointer:])

    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
        leiden_labels.append(labels_tmp[start_pointer:end_pointer])


utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = scmomat_dir + f'latent_separate_{K}_{T}_processed.png', 
                      figsize = (10,27), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = scmomat_dir + f'latent_joint_{K}_{T}_processed.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = scmomat_dir + f'latent_batches_{K}_{T}_processed.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7)

utils.plot_latent_ext(x_umaps, annos = leiden_labels, mode = "joint", save = scmomat_dir + f'latent_leiden_clusters_{K}_{T}_{resolution}_processed.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7)


# In[]
# 1. UINMF
uinmf_path = result_dir + "/uinmf/" 
H1_uinmf = pd.read_csv(uinmf_path + "H1_norm.csv", index_col = 0).values
H2_uinmf = pd.read_csv(uinmf_path + "H2_norm.csv", index_col = 0).values
H3_uinmf = pd.read_csv(uinmf_path + "H3_norm.csv", index_col = 0).values
H4_uinmf = pd.read_csv(uinmf_path + "H4_norm.csv", index_col = 0).values
H5_uinmf = pd.read_csv(uinmf_path + "H5_norm.csv", index_col = 0).values
H6_uinmf = pd.read_csv(uinmf_path + "H6_norm.csv", index_col = 0).values

uinmf_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(np.concatenate((H1_uinmf, H2_uinmf, H3_uinmf, H4_uinmf, H5_uinmf, H6_uinmf), axis = 0))
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
                      figsize = (10,27), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

utils.plot_latent_ext(uinmf_umaps, annos = labels, mode = "modality", save = uinmf_path + f'latent_batches_uinmf.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

utils.plot_latent_ext(uinmf_umaps, annos = labels, mode = "joint", save = uinmf_path + f'latent_joint_uinmf.png', 
                      figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)


# # 1. Liger
# liger_path = result_dir + "/liger/" 
# H1_liger = pd.read_csv(liger_path + "H1_norm.csv", index_col = 0).values
# H2_liger = pd.read_csv(liger_path + "H2_norm.csv", index_col = 0).values
# H3_liger = pd.read_csv(liger_path + "H3_norm.csv", index_col = 0).values
# H4_liger = pd.read_csv(liger_path + "H4_norm.csv", index_col = 0).values
# H5_liger = pd.read_csv(liger_path + "H5_norm.csv", index_col = 0).values
# H6_liger = pd.read_csv(liger_path + "H6_norm.csv", index_col = 0).values

# liger_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(np.concatenate((H1_liger, H2_liger, H3_liger, H4_liger, H5_liger, H6_liger), axis = 0))
# liger_umaps = []
# for batch in range(n_batches):
#     if batch == 0:
#         start_pointer = 0
#         end_pointer = start_pointer + zs[batch].shape[0]
#         liger_umaps.append(liger_umap[start_pointer:end_pointer,:])
#     elif batch == (n_batches - 1):
#         start_pointer = start_pointer + zs[batch - 1].shape[0]
#         liger_umaps.append(liger_umap[start_pointer:,:])
#     else:
#         start_pointer = start_pointer + zs[batch - 1].shape[0]
#         end_pointer = start_pointer + zs[batch].shape[0]
#         liger_umaps.append(liger_umap[start_pointer:end_pointer,:])

# utils.plot_latent_ext(liger_umaps, annos = labels, mode = "separate", save = liger_path + f'latent_separate_liger.png', 
#                       figsize = (10,27), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

# utils.plot_latent_ext(liger_umaps, annos = labels, mode = "modality", save = liger_path + f'latent_batches_liger.png', 
#                       figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

# utils.plot_latent_ext(liger_umaps, annos = labels, mode = "joint", save = liger_path + f'latent_joint_liger.png', 
#                       figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

# Multimap
multimap_path = result_dir + "/multimap/"
batches = pd.read_csv(multimap_path + "batch_id.csv", index_col = 0)
X_multimap = np.load(multimap_path + "multimap.npy")
G_multimap = sp.load_npz(multimap_path + "multimap_graph.npz").todense()
X_multimaps = []
for batch in ["C1", "C2", "C3", "C4", "C5", "C6"]:
    X_multimaps.append(X_multimap[batches.values.squeeze() == batch, :])


utils.plot_latent_ext(X_multimaps, annos = labels, mode = "separate", save = multimap_path + f'latent_separate_multimap.png', 
                      figsize = (10,27), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(X_multimaps, annos = labels, mode = "modality", save = multimap_path + f'latent_batches_multimap.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(X_multimaps, annos = labels, mode = "joint", save = multimap_path + f'latent_joint_multimap.png', 
                      figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7)


# In[]
import importlib 
importlib.reload(utils)
importlib.reload(bmk)
n_neighbors = 30
# graph connectivity score (gc) measure the batch effect removal per cell identity
# 1. scMoMaT
gc_scmomat = bmk.graph_connectivity(X = np.concatenate(zs, axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (scMoMaT): {:.3f}'.format(gc_scmomat))

# 2. UINMF
gc_uinmf = bmk.graph_connectivity(X = np.concatenate((H1_uinmf, H2_uinmf, H3_uinmf, H4_uinmf, H5_uinmf, H6_uinmf), axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (UINMF): {:.3f}'.format(gc_uinmf))

# # 2. LIGER
# gc_liger = bmk.graph_connectivity(X = np.concatenate((H1_liger, H2_liger, H3_liger, H4_liger, H5_liger, H6_liger), axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
# print('GC (LIGER): {:.3f}'.format(gc_liger))

# 3. Multimap
G_multimap[G_multimap == 0] = np.inf
knn_indices_multimap = G_multimap.argsort(axis = 1)[:, :n_neighbors]
knn_graph_multimap = np.zeros_like(G_multimap)
knn_graph_multimap[np.arange(knn_indices_multimap.shape[0])[:, None], knn_indices_multimap] = 1
gc_multimap = bmk.graph_connectivity(G = knn_graph_multimap, groups = np.concatenate(labels, axis = 0), k = n_neighbors)
gc_multimap2 = bmk.graph_connectivity(X = np.concatenate(X_multimaps, axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (MultiMap Graph): {:.3f}'.format(gc_multimap))
print('GC (MultiMap): {:.3f}'.format(gc_multimap2))


# Conservation of biological identity
# NMI and ARI
# 1. scMoMaT
nmi_scmomat = []
ari_scmomat = []
for resolution in np.arange(0.1, 10, 0.5):
    # leiden_labels_scjmt = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
    leiden_labels_smomat = utils.leiden_cluster(X = np.concatenate(zs, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_scmomat.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_smomat))
    ari_scmomat.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_smomat))
print('NMI (scMoMaT): {:.3f}'.format(max(nmi_scmomat)))
print('ARI (scMoMaT): {:.3f}'.format(max(ari_scmomat)))

# 2. UINMF
nmi_uinmf = []
ari_uinmf = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_uinmf = utils.leiden_cluster(X = np.concatenate((H1_uinmf, H2_uinmf, H3_uinmf, H4_uinmf, H5_uinmf, H6_uinmf), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_uinmf.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_uinmf))
    ari_uinmf.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_uinmf))
print('NMI (UINMF): {:.3f}'.format(max(nmi_uinmf)))
print('ARI (UINMF): {:.3f}'.format(max(ari_uinmf)))

# # 2. Liger
# nmi_liger = []
# ari_liger = []
# for resolution in np.arange(0.1, 10, 0.5):
#     leiden_labels_liger = utils.leiden_cluster(X = np.concatenate((H1_liger, H2_liger, H3_liger, H4_liger, H5_liger, H6_liger), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
#     nmi_liger.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_liger))
#     ari_liger.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_liger))
# print('NMI (LIGER): {:.3f}'.format(max(nmi_liger)))
# print('ARI (LIGER): {:.3f}'.format(max(ari_liger)))

# 3. Multimap
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

# scores = pd.DataFrame(columns = ["methods", "resolution", "NMI", "ARI", "GC"])
# scores["NMI"] = np.array(nmi_scmomat + nmi_uinmf + nmi_liger + nmi_multimap)
# scores["ARI"] = np.array(ari_scmomat + ari_uinmf + ari_liger + ari_multimap)
# scores["GC"] = np.array([gc_scmomat] * len(nmi_scmomat) + [gc_uinmf] * len(nmi_uinmf) +  [gc_liger] * len(nmi_liger) +[gc_multimap] * len(ari_multimap))
# scores["resolution"] = np.array([x for x in np.arange(0.1, 10, 0.5)] * 4)
# scores["methods"] = np.array(["scMoMaT"] * len(nmi_scmomat) + ["UINMF"] * len(nmi_uinmf) + ["LIGER"] * len(nmi_liger) + ["MultiMap"] * len(ari_multimap))

# NO LIGER
scores = pd.DataFrame(columns = ["methods", "resolution", "NMI", "ARI", "GC"])
scores["NMI"] = np.array(nmi_scmomat + nmi_uinmf + nmi_multimap)
scores["ARI"] = np.array(ari_scmomat + ari_uinmf + ari_multimap)
scores["GC"] = np.array([gc_scmomat] * len(nmi_scmomat) + [gc_uinmf] * len(nmi_uinmf) + [gc_multimap] * len(ari_multimap))
scores["resolution"] = np.array([x for x in np.arange(0.1, 10, 0.5)] * 3)
scores["methods"] = np.array(["scMoMaT"] * len(nmi_scmomat) + ["UINMF"] * len(nmi_uinmf) + ["MultiMap"] * len(ari_multimap))

scores.to_csv(result_dir + "score.csv")

# In[]
nmi_scmomat = []
ari_scmomat = []
gc_scmomat = []
nmi_uinmf = []
ari_uinmf = []
gc_uinmf = []
nmi_liger = []
ari_liger = []
gc_liger = []
nmi_multimap = []
ari_multimap = []
gc_multimap = []
# for seed in [1,2,3,4,9]:
for seed in [1,2,3,4,5,6,7,9]:
    result_dir = f'simulated/6b16c_{seed}_large2_1'
    scores = pd.read_csv(result_dir + "score.csv", index_col = 0)
    scores_scmomat = scores[scores["methods"] == "scMoMaT"]
    scores_uinmf = scores[scores["methods"] == "UINMF"]
    scores_liger = scores[scores["methods"] == "LIGER"]
    scores_multimap = scores[scores["methods"] == "MultiMap"]
    nmi_scmomat.append(np.max(scores_scmomat["NMI"].values))
    ari_scmomat.append(np.max(scores_scmomat["ARI"].values))
    gc_scmomat.append(np.max(scores_scmomat["GC"].values))
    nmi_uinmf.append(np.max(scores_uinmf["NMI"].values))
    ari_uinmf.append(np.max(scores_uinmf["ARI"].values))
    gc_uinmf.append(np.max(scores_uinmf["GC"].values))
    nmi_liger.append(np.max(scores_liger["NMI"].values))
    ari_liger.append(np.max(scores_liger["ARI"].values))
    gc_liger.append(np.max(scores_liger["GC"].values))
    nmi_multimap.append(np.max(scores_multimap["NMI"].values))
    ari_multimap.append(np.max(scores_multimap["ARI"].values))
    gc_multimap.append(np.max(scores_multimap["GC"].values))

new_score = pd.DataFrame()
new_score["method"] = ["scMoMaT"] * len(ari_scmomat) + ["MultiMap"] * len(ari_multimap) + ["UINMF"] * len(ari_uinmf) + ["LIGER"] * len(ari_liger)
new_score["ARI"] = ari_scmomat + ari_multimap + ari_uinmf + ari_liger
new_score["NMI"] = nmi_scmomat + nmi_multimap + nmi_uinmf + nmi_liger
new_score["GC"] = gc_scmomat + gc_multimap + gc_uinmf + gc_liger

import seaborn as sns
plt.rcParams["font.size"] = 20
fig = plt.figure(figsize = (20, 7))
ax = fig.subplots(nrows = 1, ncols = 3)
sns.boxplot(data = new_score, x = "method", y = "GC", ax = ax[0])
sns.boxplot(data = new_score, x = "method", y = "ARI", ax = ax[1])
sns.boxplot(data = new_score, x = "method", y = "NMI", ax = ax[2])
ax[0].set_title("Graph connectivity")
ax[1].set_title("ARI")
ax[2].set_title("NMI")
fig.tight_layout()
fig.savefig("simulated/scores1.png", bbox_inches = "tight")
    
# In[]
nmi_scmomat = []
ari_scmomat = []
gc_scmomat = []
nmi_uinmf = []
ari_uinmf = []
gc_uinmf = []
nmi_liger = []
ari_liger = []
gc_liger = []
nmi_multimap = []
ari_multimap = []
gc_multimap = []
# for seed in [1,2,3,4,9]:
for seed in [1,2,3,4,5,6,7,9]:
    result_dir = f'simulated/6b16c_{seed}_large2_2'
    scores = pd.read_csv(result_dir + "score.csv", index_col = 0)
    scores_scmomat = scores[scores["methods"] == "scMoMaT"]
    scores_uinmf = scores[scores["methods"] == "UINMF"]
    scores_multimap = scores[scores["methods"] == "MultiMap"]
    nmi_scmomat.append(np.max(scores_scmomat["NMI"].values))
    ari_scmomat.append(np.max(scores_scmomat["ARI"].values))
    gc_scmomat.append(np.max(scores_scmomat["GC"].values))
    nmi_uinmf.append(np.max(scores_uinmf["NMI"].values))
    ari_uinmf.append(np.max(scores_uinmf["ARI"].values))
    gc_uinmf.append(np.max(scores_uinmf["GC"].values))
    nmi_multimap.append(np.max(scores_multimap["NMI"].values))
    ari_multimap.append(np.max(scores_multimap["ARI"].values))
    gc_multimap.append(np.max(scores_multimap["GC"].values))

new_score = pd.DataFrame()
new_score["method"] = ["scMoMaT"] * len(ari_scmomat) + ["MultiMap"] * len(ari_multimap) + ["UINMF"] * len(ari_uinmf)
new_score["ARI"] = ari_scmomat + ari_multimap + ari_uinmf
new_score["NMI"] = nmi_scmomat + nmi_multimap + nmi_uinmf
new_score["GC"] = gc_scmomat + gc_multimap + gc_uinmf

import seaborn as sns
plt.rcParams["font.size"] = 20
fig = plt.figure(figsize = (20, 7))
ax = fig.subplots(nrows = 1, ncols = 3)
sns.boxplot(data = new_score, x = "method", y = "GC", ax = ax[0])
sns.boxplot(data = new_score, x = "method", y = "ARI", ax = ax[1])
sns.boxplot(data = new_score, x = "method", y = "NMI", ax = ax[2])
ax[0].set_title("Graph connectivity")
ax[1].set_title("ARI")
ax[2].set_title("NMI")
fig.tight_layout()
fig.savefig("simulated/scores2.png", bbox_inches = "tight")

# %%
