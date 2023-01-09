# In[]
import sys, os
sys.path.append('../')

import numpy as np
from umap import UMAP
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd  
import scipy.sparse as sp

import scmomat.model as model
import scmomat.utils as utils
import scmomat.bmk as bmk
import scmomat.umap_batch as umap_batch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

# # In[]
# # ------------------------------------------------------------------------------------------------------------------------------------------------------
# #
# #   NOTE: 1. Subsampling the data batches to create imbalanced datasets 
# #
# # ------------------------------------------------------------------------------------------------------------------------------------------------------
# # NOTE: read in dataset
# data_dir = "../data/simulated/6b16c_test_10/unequal/"
# # subsample batches 0, 3, 5 by 10: [::10]
# imbalanced_dir = "../data/simulated/6b16c_test_10/imbalanced/"

# if not os.path.exists(imbalanced_dir):
#     os.makedirs(imbalanced_dir)

# n_batches = 6
# for batch in range(n_batches):        
#     label = pd.read_csv(os.path.join(data_dir, 'cell_label' + str(batch + 1) + '.txt'), index_col=0, sep = "\t")
#     if batch in [0,3,5]:
#         # subsample by 10
#         label = label.iloc[::10,:]
#     label.to_csv(os.path.join(imbalanced_dir, 'cell_label' + str(batch + 1) + '.txt'), sep = "\t")
#     print("number of cells: {:d}".format(label.shape[0]))

#     counts_atac = np.loadtxt(os.path.join(data_dir, 'RxC' + str(batch + 1) + ".txt"), delimiter = "\t")
#     if batch in [0,3,5]:
#         # subsample by 10
#         counts_atac = counts_atac[:,::10]
#     np.savetxt(os.path.join(imbalanced_dir, 'RxC' + str(batch + 1) + '.txt'), X = counts_atac, delimiter = "\t")

#     counts_rna = np.loadtxt(os.path.join(data_dir, 'GxC' + str(batch + 1) + ".txt"), delimiter = "\t")
#     if batch in [0,3,5]:
#         # subsample by 10
#         counts_rna = counts_rna[:,::10]
#     np.savetxt(os.path.join(imbalanced_dir, 'GxC' + str(batch + 1) + '.txt'), X = counts_rna, delimiter = "\t")
    
# A = np.loadtxt(os.path.join(data_dir +'region2gene.txt'), delimiter = "\t")
# np.savetxt(os.path.join(imbalanced_dir, 'region2gene.txt'), X = A, delimiter = "\t")
# In[]
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   NOTE: 1. Load dataset and running scmomat (without retraining, retraining see the third section)
#
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# NOTE: read in dataset
data_dir = "../data/simulated/6b16c_test_1/imbalanced/"
result_dir = "simulated/6b16c_test_1/imbalanced"
scmomat_dir = result_dir + "/scmomat/"

if not os.path.exists(scmomat_dir):
    os.makedirs(scmomat_dir)

n_batches = 6
counts_rnas = []
counts_atacs = []
labels = []
for batch in range(n_batches):        
    label = pd.read_csv(os.path.join(data_dir, 'cell_label' + str(batch + 1) + '.txt'), index_col=0, sep = "\t")["pop"].values.squeeze()
    labels.append(label)
    print("number of cells: {:d}".format(label.shape[0]))
    try:
        counts_atac = np.loadtxt(os.path.join(data_dir, 'RxC' + str(batch + 1) + ".txt"), delimiter = "\t").T
        counts_atac = utils.preprocess(counts_atac, modality = "ATAC")
        print("read atac for batch" + str(batch + 1))
    except:
        counts_atac = None
        
    try:
        counts_rna = np.loadtxt(os.path.join(data_dir, 'GxC' + str(batch + 1) + ".txt"), delimiter = "\t").T
        print("read rna for batch" + str(batch + 1))
        counts_rna = utils.preprocess(counts_rna, modality = "RNA", log = False)
    except:
        counts_rna = None
    
    # preprocess the count matrix
    counts_rnas.append(counts_rna)
    counts_atacs.append(counts_atac)

counts = {"rna":counts_rnas, "atac": counts_atacs}


# NOTE: SCENARIO 1: diagonal integration
counts["rna"][0] = None
counts["rna"][1] = None
counts["rna"][2] = None
# counts["atac"][3] = None 
counts["atac"][4] = None
counts["atac"][5] = None

# No need for pseudo-count matrix
A = np.loadtxt(os.path.join(data_dir +'region2gene.txt'), delimiter = "\t").T

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


# In[]
# NOTE: Running scmomat
# weight on regularization term
lamb = 0.001
batchsize = 0.1
# running seed
seed = 0
# number of latent dimensions
K = 20
interval = 1000
T = 4000
lr = 1e-2

# start_time = time.time()
# model1 = model.scmomat_model(counts = counts, K = K, batch_size = batchsize, interval = interval, lr = lr, lamb = lamb, seed = seed, device = device)
# losses1 = model1.train_func(T = T)
# end_time = time.time()
# print("running time: " + str(end_time - start_time))

# torch.save(model1, scmomat_dir + f'CFRM_{K}_{T}.pt')
model1 = torch.load(scmomat_dir + f'CFRM_{K}_{T}.pt')


# In[]
# NOTE: Plot the result before post-processing
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
# NOTE: Post-processing, clustering, and plot the result after post-processing
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
                      figsize = (7,20), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = scmomat_dir + f'latent_joint_{K}_{T}_processed.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = scmomat_dir + f'latent_batches_{K}_{T}_processed.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, alpha = 0.7)

utils.plot_latent_ext(x_umaps, annos = leiden_labels, mode = "joint", save = scmomat_dir + f'latent_leiden_clusters_{K}_{T}_{resolution}_processed.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, alpha = 0.7)


# In[]
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   NOTE: 2. Benchmarking with baseline methods
#
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# NOTE: Baseline methods
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
                      figsize = (7,20), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", alpha = 0.7)

utils.plot_latent_ext(uinmf_umaps, annos = labels, mode = "modality", save = uinmf_path + f'latent_batches_uinmf.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, text_size = "large", alpha = 0.7)

utils.plot_latent_ext(uinmf_umaps, annos = labels, mode = "joint", save = uinmf_path + f'latent_joint_uinmf.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, text_size = "large", alpha = 0.7)


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
G_multimap = sp.load_npz(multimap_path + "multimap_graph.npz").toarray()
X_multimaps = []
for batch in ["C1", "C2", "C3", "C4", "C5", "C6"]:
    X_multimaps.append(X_multimap[batches.values.squeeze() == batch, :])


utils.plot_latent_ext(X_multimaps, annos = labels, mode = "separate", save = multimap_path + f'latent_separate_multimap.png', 
                      figsize = (7,20), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(X_multimaps, annos = labels, mode = "modality", save = multimap_path + f'latent_batches_multimap.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(X_multimaps, annos = labels, mode = "joint", save = multimap_path + f'latent_joint_multimap.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, alpha = 0.7)


# Stabmap
stabmap_path = result_dir + "/stabmap/"
stabmap_b1 = pd.read_csv(stabmap_path + "stab_b1.csv", index_col = 0).values
stabmap_b2 = pd.read_csv(stabmap_path + "stab_b2.csv", index_col = 0).values
stabmap_b3 = pd.read_csv(stabmap_path + "stab_b3.csv", index_col = 0).values
stabmap_b4 = pd.read_csv(stabmap_path + "stab_b4.csv", index_col = 0).values
stabmap_b5 = pd.read_csv(stabmap_path + "stab_b5.csv", index_col = 0).values
stabmap_b6 = pd.read_csv(stabmap_path + "stab_b6.csv", index_col = 0).values

stabmap_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(np.concatenate((stabmap_b1, stabmap_b2, stabmap_b3, stabmap_b4, stabmap_b5, stabmap_b6), axis = 0))
stabmap_umaps = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        stabmap_umaps.append(stabmap_umap[start_pointer:end_pointer,:])
    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        stabmap_umaps.append(stabmap_umap[start_pointer:,:])
    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        stabmap_umaps.append(stabmap_umap[start_pointer:end_pointer,:])

utils.plot_latent_ext(stabmap_umaps, annos = labels, mode = "separate", save = stabmap_path + f'latent_separate_stabmap.png', 
                      figsize = (7,20), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", alpha = 0.7)

utils.plot_latent_ext(stabmap_umaps, annos = labels, mode = "modality", save = stabmap_path + f'latent_batches_stabmap.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, text_size = "large", alpha = 0.7)

utils.plot_latent_ext(stabmap_umaps, annos = labels, mode = "joint", save = stabmap_path + f'latent_joint_stabmap.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, text_size = "large", alpha = 0.7)


# In[]
n_neighbors =  knn_indices.shape[1]
# graph connectivity score (gc) measure the batch effect removal per cell identity
# 1. scMoMaT
knn_graph = np.zeros((knn_indices.shape[0], knn_indices.shape[0]))
knn_graph[np.arange(knn_indices.shape[0])[:, None], knn_indices] = 1
gc_scmomat = bmk.graph_connectivity(G = knn_graph, groups = np.concatenate(labels, axis = 0))
print('GC (scmomat): {:.3f}'.format(gc_scmomat))

# 2. UINMF
gc_uinmf = bmk.graph_connectivity(X = np.concatenate((H1_uinmf, H2_uinmf, H3_uinmf, H4_uinmf, H5_uinmf, H6_uinmf), axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (UINMF): {:.3f}'.format(gc_uinmf))

# # 2. LIGER
# gc_liger = bmk.graph_connectivity(X = np.concatenate((H1_liger, H2_liger, H3_liger, H4_liger, H5_liger, H6_liger), axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
# print('GC (LIGER): {:.3f}'.format(gc_liger))

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

# 4. Stabmap
gc_stabmap = bmk.graph_connectivity(X = np.concatenate((stabmap_b1, stabmap_b2, stabmap_b3, stabmap_b4, stabmap_b5, stabmap_b6), axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (Stabmap): {:.3f}'.format(gc_stabmap))

# Conservation of biological identity
# NMI, ARI, and F1

# F1 score: rare cell type detection
gt_labels = np.concatenate(labels)
uniq_labels, label_counts = np.unique(gt_labels, return_counts = True)
rare_label = uniq_labels[np.argsort(label_counts)[0]]
gt_rare_labels = np.where(gt_labels == rare_label, 1, 0)

# 1. scMoMaT
nmi_scmomat = []
ari_scmomat = []
f1_scmomat = []
for resolution in np.arange(0.1, 10, 0.5):
    # use the post-processed graph
    leiden_labels_scmomat = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
    nmi_scmomat.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_scmomat))
    ari_scmomat.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_scmomat))
    # calculate F1 score
    uniq_labels, label_counts = np.unique(leiden_labels_scmomat[np.where(gt_labels == rare_label)[0]], return_counts = True)
    predict_rare_label = uniq_labels[np.argsort(label_counts)[-1]]
    predict_rare_labels = np.where(leiden_labels_scmomat == predict_rare_label, 1, 0) 
    f1_scmomat.append(bmk.F1_score(gt_rare_labels, predict_rare_labels))

print('NMI (scMoMaT): {:.3f}'.format(max(nmi_scmomat)))
print('ARI (scMoMaT): {:.3f}'.format(max(ari_scmomat)))
print('F1 (scMoMaT): {:.3f}'.format(max(f1_scmomat)))


# 2. UINMF
nmi_uinmf = []
ari_uinmf = []
f1_uinmf = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_uinmf = utils.leiden_cluster(X = np.concatenate((H1_uinmf, H2_uinmf, H3_uinmf, H4_uinmf, H5_uinmf, H6_uinmf), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_uinmf.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_uinmf))
    ari_uinmf.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_uinmf))
    # calculate F1 score
    uniq_labels, label_counts = np.unique(leiden_labels_uinmf[np.where(gt_labels == rare_label)[0]], return_counts = True)
    predict_rare_label = uniq_labels[np.argsort(label_counts)[-1]]
    predict_rare_labels = np.where(leiden_labels_uinmf == predict_rare_label, 1, 0) 
    f1_uinmf.append(bmk.F1_score(gt_rare_labels, predict_rare_labels))

print('NMI (UINMF): {:.3f}'.format(max(nmi_uinmf)))
print('ARI (UINMF): {:.3f}'.format(max(ari_uinmf)))
print('F1 (UINMF): {:.3f}'.format(max(f1_uinmf)))


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
G_multimap = sp.load_npz(multimap_path + "multimap_graph.npz").toarray()
nmi_multimap = []
ari_multimap = []
f1_multimap = []
for resolution in np.arange(0.1, 10, 0.5):
    # leiden_labels_seurat = utils.leiden_cluster(X = np.concatenate(seurat_pcas, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    # Multimap state to use graph for clustering, leiden cluster the same as multimap tutorial [Checked]
    leiden_labels_multimap = utils.leiden_cluster(affin = G_multimap, resolution = resolution)
    nmi_multimap.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_multimap))
    ari_multimap.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_multimap))
    # calculate F1 score
    uniq_labels, label_counts = np.unique(leiden_labels_multimap[np.where(gt_labels == rare_label)[0]], return_counts = True)
    predict_rare_label = uniq_labels[np.argsort(label_counts)[-1]]
    predict_rare_labels = np.where(leiden_labels_multimap == predict_rare_label, 1, 0) 
    f1_multimap.append(bmk.F1_score(gt_rare_labels, predict_rare_labels))

print('NMI (MultiMap): {:.3f}'.format(max(nmi_multimap)))
print('ARI (MultiMap): {:.3f}'.format(max(ari_multimap)))
print('F1 (MultiMap): {:.3f}'.format(max(f1_multimap)))

# 4. Stabmap
nmi_stabmap = []
ari_stabmap = []
f1_stabmap = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_stabmap = utils.leiden_cluster(X = np.concatenate((stabmap_b1, stabmap_b2, stabmap_b3, stabmap_b4, stabmap_b5, stabmap_b6), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_stabmap.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_stabmap))
    ari_stabmap.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_stabmap))
    # calculate F1 score
    uniq_labels, label_counts = np.unique(leiden_labels_stabmap[np.where(gt_labels == rare_label)[0]], return_counts = True)
    predict_rare_label = uniq_labels[np.argsort(label_counts)[-1]]
    predict_rare_labels = np.where(leiden_labels_stabmap == predict_rare_label, 1, 0) 
    f1_stabmap.append(bmk.F1_score(gt_rare_labels, predict_rare_labels))

print('NMI (Stabmap): {:.3f}'.format(max(nmi_stabmap)))
print('ARI (Stabmap): {:.3f}'.format(max(ari_stabmap)))
print('F1 (Stabmap): {:.3f}'.format(max(f1_stabmap)))

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

# UINMF
lta_uinmf = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, 
                                  z_query = np.concatenate((H1_uinmf, H2_uinmf, H3_uinmf, H4_uinmf, H5_uinmf, H6_uinmf), axis = 0)[query_cell,:],
                                  z_train = np.concatenate((H1_uinmf, H2_uinmf, H3_uinmf, H4_uinmf, H5_uinmf, H6_uinmf), axis = 0)[training_cell,:])

# MultiMap
G_multimap = sp.load_npz(multimap_path + "multimap_graph.npz").toarray()
knn_indices_multimap = G_multimap.argsort(axis = 1)[:, -n_neighbors:]
knn_graph_multimap = np.zeros_like(G_multimap)
knn_graph_multimap[np.arange(knn_indices_multimap.shape[0])[:, None], knn_indices_multimap] = 1
lta_multimap = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, knn_graph = knn_graph_multimap[query_cell, :][:, training_cell])
lta_multimap2 = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, 
                                  z_query = np.concatenate(X_multimaps, axis = 0)[query_cell,:],
                                  z_train = np.concatenate(X_multimaps, axis = 0)[training_cell,:])

# stabmap
lta_stabmap = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, 
                                  z_query = np.concatenate((stabmap_b1, stabmap_b2, stabmap_b3, stabmap_b4, stabmap_b5, stabmap_b6), axis = 0)[query_cell,:],
                                  z_train = np.concatenate((stabmap_b1, stabmap_b2, stabmap_b3, stabmap_b4, stabmap_b5, stabmap_b6), axis = 0)[training_cell,:])


print("Label transfer accuracy (scMoMaT): {:.3f}".format(lta_scmomat))
print("Label transfer accuracy (UINMF): {:.3f}".format(lta_uinmf))
print("Label transfer accuracy (MultiMap Graph): {:.3f}".format(lta_multimap))
print("Label transfer accuracy (MultiMap): {:.3f}".format(lta_multimap2))
print("Label transfer accuracy (Stabmap): {:.3f}".format(lta_stabmap))

# scores = pd.DataFrame(columns = ["methods", "resolution", "NMI", "ARI", "GC"])
# scores["NMI"] = np.array(nmi_scmomat + nmi_uinmf + nmi_liger + nmi_multimap)
# scores["ARI"] = np.array(ari_scmomat + ari_uinmf + ari_liger + ari_multimap)
# scores["GC"] = np.array([gc_scmomat] * len(nmi_scmomat) + [gc_uinmf] * len(nmi_uinmf) +  [gc_liger] * len(nmi_liger) +[gc_multimap] * len(ari_multimap))
# scores["resolution"] = np.array([x for x in np.arange(0.1, 10, 0.5)] * 4)
# scores["methods"] = np.array(["scMoMaT"] * len(nmi_scmomat) + ["UINMF"] * len(nmi_uinmf) + ["LIGER"] * len(nmi_liger) + ["MultiMap"] * len(ari_multimap))

# NO LIGER
scores = pd.DataFrame(columns = ["methods", "resolution", "NMI", "ARI", "GC", "LTA", "F1"])
scores["NMI"] = np.array(nmi_scmomat + nmi_uinmf + nmi_multimap + nmi_stabmap)
scores["ARI"] = np.array(ari_scmomat + ari_uinmf + ari_multimap + ari_stabmap)
scores["F1"] = np.array(f1_scmomat + f1_uinmf + f1_multimap + f1_stabmap)
scores["GC"] = np.array([gc_scmomat] * len(nmi_scmomat) + [gc_uinmf] * len(nmi_uinmf) + [gc_multimap] * len(ari_multimap) + [gc_stabmap] * len(ari_stabmap))
scores["LTA"] = np.array([lta_scmomat] * len(nmi_scmomat) + [lta_uinmf] * len(nmi_uinmf) + [lta_multimap] * len(ari_multimap) + [lta_stabmap] * len(ari_stabmap))
scores["resolution"] = np.array([x for x in np.arange(0.1, 10, 0.5)] * 4)
scores["methods"] = np.array(["scMoMaT"] * len(nmi_scmomat) + ["UINMF"] * len(nmi_uinmf) + ["MultiMap"] * len(ari_multimap) + ["Stabmap"] * len(ari_stabmap))

scores.to_csv(result_dir + "/score.csv")



# In[]
if True:
    nmi_scmomat = []
    ari_scmomat = []
    gc_scmomat = []
    lta_scmomat = []
    f1_scmomat = []
    nmi_uinmf = []
    ari_uinmf = []
    gc_uinmf = []
    lta_uinmf = []
    f1_uinmf = []
    nmi_liger = []
    ari_liger = []
    gc_liger = []
    lta_liger = []
    f1_liger = []
    nmi_multimap = []
    ari_multimap = []
    gc_multimap = []
    lta_multimap = []
    f1_multimap = []
    nmi_stabmap = []
    ari_stabmap = []
    gc_stabmap = []
    lta_stabmap = []
    f1_stabmap = []
    # for seed in [1,2,3,4,9]:
    # ARI higher: 2, 3, 9
    for seed in [1,2,3,4,5,6,7,9]:
        result_dir = f'simulated/6b16c_test_{seed}/imbalanced/'
        scores = pd.read_csv(result_dir + "score.csv", index_col = 0)
        scores_scmomat = scores[scores["methods"] == "scMoMaT"]
        scores_uinmf = scores[scores["methods"] == "UINMF"]
        # scores_liger = scores[scores["methods"] == "LIGER"]
        scores_multimap = scores[scores["methods"] == "MultiMap"]
        scores_stabmap = scores[scores["methods"] == "Stabmap"]

        nmi_scmomat.append(np.max(scores_scmomat["NMI"].values))
        ari_scmomat.append(np.max(scores_scmomat["ARI"].values))
        gc_scmomat.append(np.max(scores_scmomat["GC"].values))
        lta_scmomat.append(np.max(scores_scmomat["LTA"].values))
        f1_scmomat.append(np.max(scores_scmomat["F1"].values))

        nmi_uinmf.append(np.max(scores_uinmf["NMI"].values))
        ari_uinmf.append(np.max(scores_uinmf["ARI"].values))
        gc_uinmf.append(np.max(scores_uinmf["GC"].values))
        lta_uinmf.append(np.max(scores_uinmf["LTA"].values))
        f1_uinmf.append(np.max(scores_uinmf["F1"].values))

        # nmi_liger.append(np.max(scores_liger["NMI"].values))
        # ari_liger.append(np.max(scores_liger["ARI"].values))
        # gc_liger.append(np.max(scores_liger["GC"].values))
        # lta_liger.append(np.max(scores_liger["LTA"].values))
        
        nmi_multimap.append(np.max(scores_multimap["NMI"].values))
        ari_multimap.append(np.max(scores_multimap["ARI"].values))
        gc_multimap.append(np.max(scores_multimap["GC"].values))
        lta_multimap.append(np.max(scores_multimap["LTA"].values))
        f1_multimap.append(np.max(scores_multimap["F1"].values))

        nmi_stabmap.append(np.max(scores_stabmap["NMI"].values))
        ari_stabmap.append(np.max(scores_stabmap["ARI"].values))
        gc_stabmap.append(np.max(scores_stabmap["GC"].values))
        lta_stabmap.append(np.max(scores_stabmap["LTA"].values))
        f1_stabmap.append(np.max(scores_stabmap["F1"].values))

    new_score = pd.DataFrame()
    new_score["method"] = ["scMoMaT"] * len(ari_scmomat) + ["MultiMap"] * len(ari_multimap) + ["UINMF"] * len(ari_uinmf) + ["Stabmap"] * len(ari_stabmap)
    new_score["ARI"] = ari_scmomat + ari_multimap + ari_uinmf + ari_stabmap
    new_score["NMI"] = nmi_scmomat + nmi_multimap + nmi_uinmf + nmi_stabmap
    new_score["GC"] = gc_scmomat + gc_multimap + gc_uinmf + gc_stabmap
    new_score["LTA"] = lta_scmomat + lta_multimap + lta_uinmf + lta_stabmap
    new_score["F1"] = f1_scmomat + f1_multimap + f1_uinmf + f1_stabmap


    import seaborn as sns
    plt.rcParams["font.size"] = 20
    fig = plt.figure(figsize = (32, 5))
    ax = fig.subplots(nrows = 1, ncols = 5)
    sns.boxplot(data = new_score, x = "method", y = "GC", ax = ax[0])
    sns.stripplot(data = new_score, x = "method", y = "GC", ax = ax[0], color = "black")
    sns.boxplot(data = new_score, x = "method", y = "ARI", ax = ax[1])
    sns.stripplot(data = new_score, x = "method", y = "ARI", ax = ax[1], color = "black")
    sns.boxplot(data = new_score, x = "method", y = "NMI", ax = ax[2])
    sns.stripplot(data = new_score, x = "method", y = "NMI", ax = ax[2], color = "black")    
    sns.boxplot(data = new_score, x = "method", y = "LTA", ax = ax[3])
    sns.stripplot(data = new_score, x = "method", y = "LTA", ax = ax[3], color = "black")
    sns.boxplot(data = new_score, x = "method", y = "F1", ax = ax[4])
    sns.stripplot(data = new_score, x = "method", y = "F1", ax = ax[4], color = "black")
    ax[0].set_title("Graph connectivity")
    ax[1].set_title("ARI")
    ax[2].set_title("NMI")
    ax[3].set_title("Lable Transfer Accuracy")
    ax[4].set_title("Rare cell type detection")
    fig.tight_layout()
    fig.savefig("simulated/scores_imbalanced.png", bbox_inches = "tight")
        
# %%
