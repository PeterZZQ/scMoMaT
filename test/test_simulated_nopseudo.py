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

# In[]
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   NOTE: 1. Load dataset and running scmomat (without retraining, retraining see the third section)
#
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# NOTE: read in dataset
dir = "../data/simulated/6b16c_test_1/unequal/"
result_dir = "simulated/nopseudo/"
scmomat_dir = result_dir

if not os.path.exists(scmomat_dir):
    os.makedirs(scmomat_dir)

n_batches = 3
counts_rnas = []
counts_atacs = []
labels = []
for batch in range(n_batches):        
    label = pd.read_csv(os.path.join(dir, 'cell_label' + str(batch + 1) + '.txt'), index_col=0, sep = "\t")["pop"].values.squeeze()
    labels.append(label)
    print("number of cells: {:d}".format(label.shape[0]))
    try:
        counts_atac = np.loadtxt(os.path.join(dir, 'RxC' + str(batch + 1) + ".txt"), delimiter = "\t").T
        counts_atac = utils.preprocess(counts_atac, modality = "ATAC")
        print("read atac for batch" + str(batch + 1))
    except:
        counts_atac = None
        
    try:
        counts_rna = np.loadtxt(os.path.join(dir, 'GxC' + str(batch + 1) + ".txt"), delimiter = "\t").T
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
counts["atac"][1] = None


# No need for pseudo-count matrix
A = np.loadtxt(os.path.join(dir, 'region2gene.txt'), delimiter = "\t").T

# obtain the feature name
genes = np.array(["gene_" + str(x) for x in range(counts["rna"][-1].shape[1])])
regions = np.array(["region_" + str(x) for x in range(counts["atac"][0].shape[1])])

feats_name = {"rna": genes, "atac": regions}
counts["feats_name"] = feats_name

counts["nbatches"] = n_batches

print(np.unique(np.concatenate(labels), return_counts = True))


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
# model1 = model.scmomat(counts = counts, K = K, batch_size = batchsize, interval = interval, lr = lr, lamb = lamb, seed = seed, device = device)
# losses1 = model1.train_func(T = T)
# end_time = time.time()
# print("running time: " + str(end_time - start_time))

# x = np.linspace(0, T, int(T/interval)+1)
# plt.plot(x, losses1)
# plt.yscale("log")

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
                      figsize = (7,12), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = scmomat_dir + f'latent_joint_{K}_{T}_processed.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = scmomat_dir + f'latent_batches_{K}_{T}_processed.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, alpha = 0.7)

utils.plot_latent_ext(x_umaps, annos = leiden_labels, mode = "joint", save = scmomat_dir + f'latent_leiden_clusters_{K}_{T}_{resolution}_processed.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, alpha = 0.7)

# In[]
n_neighbors =  knn_indices.shape[1]
# graph connectivity score (gc) measure the batch effect removal per cell identity
# 1. scMoMaT
knn_graph = np.zeros((knn_indices.shape[0], knn_indices.shape[0]))
knn_graph[np.arange(knn_indices.shape[0])[:, None], knn_indices] = 1
gc_scmomat = bmk.graph_connectivity(G = knn_graph, groups = np.concatenate(labels, axis = 0))
print('GC (scMoMaT): {:.3f}'.format(gc_scmomat))


# Conservation of biological identity
# NMI and ARI
# 1. scMoMaT
nmi_scmomat = []
ari_scmomat = []
for resolution in np.arange(0.1, 10, 0.5):
    # use the post-processed graph
    leiden_labels_smomat = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
    nmi_scmomat.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_smomat))
    ari_scmomat.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_smomat))
print('NMI (scMoMaT): {:.3f}'.format(max(nmi_scmomat)))
print('ARI (scMoMaT): {:.3f}'.format(max(ari_scmomat)))


# In[]
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   NOTE: 2. Add noise to atac seq in paired batch
#
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# NOTE: read in dataset
np.random.seed(0)
n_batches = 3
counts_rnas = []
counts_atacs = []
labels = []
for batch in range(n_batches):        
    label = pd.read_csv(os.path.join(dir, 'cell_label' + str(batch + 1) + '.txt'), index_col=0, sep = "\t")["pop"].values.squeeze()
    labels.append(label)
    print("number of cells: {:d}".format(label.shape[0]))
    try:
        counts_atac = np.loadtxt(os.path.join(dir, 'RxC' + str(batch + 1) + ".txt"), delimiter = "\t").T
        if batch == 2:
            # add noise, with 0.1 probability to flip the counts
            counts_atac = counts_atac * np.random.binomial(n = 1, p = 0.2, size = counts_atac.shape)
            x_lsi = lsi(counts_atac)
            x_umap = UMAP(n_components = 2, min_dist = 0.1, random_state = 0).fit_transform(x_lsi)
            print("umap atac for batch" + str(batch + 1))
            utils.plot_latent_ext([x_umap], annos = [labels[-1]], mode = "joint", save = scmomat_dir + f"RxC{batch}_noisy.png", figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True)

        counts_atac = utils.preprocess(counts_atac, modality = "ATAC")
        print("read atac for batch" + str(batch + 1))
    except:
        counts_atac = None
        
    try:
        counts_rna = np.loadtxt(os.path.join(dir, 'GxC' + str(batch + 1) + ".txt"), delimiter = "\t").T
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
counts["atac"][1] = None

# No need for pseudo-count matrix
A = np.loadtxt(os.path.join(dir, 'region2gene.txt'), delimiter = "\t").T

# obtain the feature name
genes = np.array(["gene_" + str(x) for x in range(counts["rna"][-1].shape[1])])
regions = np.array(["region_" + str(x) for x in range(counts["atac"][0].shape[1])])

feats_name = {"rna": genes, "atac": regions}
counts["feats_name"] = feats_name

counts["nbatches"] = n_batches

print(np.unique(np.concatenate(labels), return_counts = True))

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
# model1 = model.scmomat(counts = counts, K = K, batch_size = batchsize, interval = interval, lr = lr, lamb = lamb, seed = seed, device = device)
# losses1 = model1.train_func(T = T)
# end_time = time.time()
# print("running time: " + str(end_time - start_time))

# x = np.linspace(0, T, int(T/interval)+1)
# plt.plot(x, losses1)
# plt.yscale("log")

# torch.save(model1, scmomat_dir + f'CFRM_{K}_{T}_noisy.pt')
model1 = torch.load(scmomat_dir + f'CFRM_{K}_{T}_noisy.pt')


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

utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = scmomat_dir + f'latent_separate_{K}_{T}_noisy.png', figsize = (15,30), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True)
utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = scmomat_dir + f'latent_batches_{K}_{T}_noisy.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True)

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


utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = scmomat_dir + f'latent_separate_{K}_{T}_noisy_processed.png', 
                      figsize = (7,12), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = scmomat_dir + f'latent_joint_{K}_{T}_noisy_processed.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = scmomat_dir + f'latent_batches_{K}_{T}_noisy_processed.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, alpha = 0.7)

utils.plot_latent_ext(x_umaps, annos = leiden_labels, mode = "joint", save = scmomat_dir + f'latent_leiden_clusters_{K}_{T}_{resolution}_noisy_processed.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, alpha = 0.7)

# In[]
n_neighbors =  knn_indices.shape[1]
# graph connectivity score (gc) measure the batch effect removal per cell identity
# 1. scMoMaT
knn_graph = np.zeros((knn_indices.shape[0], knn_indices.shape[0]))
knn_graph[np.arange(knn_indices.shape[0])[:, None], knn_indices] = 1
gc_scmomat_noisy = bmk.graph_connectivity(G = knn_graph, groups = np.concatenate(labels, axis = 0))
print('GC (scMoMaT): {:.3f}'.format(gc_scmomat_noisy))


# Conservation of biological identity
# NMI and ARI
# 1. scMoMaT
nmi_scmomat_noisy = []
ari_scmomat_noisy = []
for resolution in np.arange(0.1, 10, 0.5):
    # use the post-processed graph
    leiden_labels_smomat = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
    nmi_scmomat_noisy.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_smomat))
    ari_scmomat_noisy.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_smomat))
print('NMI (scMoMaT): {:.3f}'.format(max(nmi_scmomat_noisy)))
print('ARI (scMoMaT): {:.3f}'.format(max(ari_scmomat_noisy)))

# %%
