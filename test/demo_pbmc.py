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
import bmk
from scipy.io import mmwrite

import model
import utils
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# In[]
dir = "../data/real/ASAP-PBMC/"
result_dir = "pbmc/scmomat/"
seurat_path = "pbmc/seurat/"
liger_path = "pbmc/liger/"

n_batches = 4
counts_rnas = []
counts_atacs = []
counts_proteins = []
labels = []
prec_labels = []
for batch in range(n_batches):
    labels.append(pd.read_csv(os.path.join(dir, 'meta_c' + str(batch + 1) + '.csv'), index_col=0)["coarse_cluster"].values.squeeze())
    prec_labels.append(pd.read_csv(os.path.join(dir, 'meta_c' + str(batch + 1) + '.csv'), index_col=0)["cluster"].values.squeeze())
    try:
        # counts_atac = sp.load_npz(os.path.join(dir, 'RxC' + str(batch + 1) + ".npz"))
        # mmwrite(os.path.join(dir, 'RxC' + str(batch + 1) + ".mtx"), counts_atac)
        counts_atac = np.array(sp.load_npz(os.path.join(dir, 'RxC' + str(batch + 1) + ".npz")).todense().T)
        counts_atac = utils.preprocess(counts_atac, modality = "ATAC")
    except:
        counts_atac = None
        
    try:
        # counts_rna = sp.load_npz(os.path.join(dir, 'GxC' + str(batch + 1) + ".npz"))
        # mmwrite(os.path.join(dir, 'GxC' + str(batch + 1) + ".mtx"), counts_rna)
        counts_rna = np.array(sp.load_npz(os.path.join(dir, 'GxC' + str(batch + 1) + ".npz")).todense().T)
        counts_rna = utils.preprocess(counts_rna, modality = "RNA", log = False)
    except:
        counts_rna = None
    
    try:
        # counts_protein = sp.load_npz(os.path.join(dir, 'PxC' + str(batch + 1) + ".npz"))
        # mmwrite(os.path.join(dir, 'PxC' + str(batch + 1) + ".mtx"), counts_protein)
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
# A1 = utils.preprocess(A1, modality = "interaction")
# A2 = utils.preprocess(A2, modality = "interaction")
# No need for pseudo-count matrix
# interacts = {"rna_atac": A2, "rna_protein": A1}

# obtain the feature name
genes = pd.read_csv(dir + "genes.txt", header = None).values.squeeze()
regions = pd.read_csv(dir + "regions.txt", header = None).values.squeeze()
proteins = pd.read_csv(dir + "proteins.txt", header = None).values.squeeze()

feats_name = {"rna": genes, "atac": regions, "protein": proteins}
counts["feats_name"] = feats_name

counts["nbatches"] = n_batches
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

# start_time = time.time()
# # model1 = model.cfrm_vanilla(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
# model1 = model.cfrm_vanilla(counts = counts, K = K, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run, device = device).to(device)
# losses1 = model1.train_func(T = T)
# end_time = time.time()
# print("running time: " + str(end_time - start_time))

# x = np.linspace(0, T, int(T/interval)+1)
# plt.plot(x, losses1)

# torch.save(model1, result_dir + f'CFRM_{K}_{T}.pt')
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
umap_op = UMAP(n_components = 2, n_neighbors = 30, min_dist = 0.2, random_state = 0) 
zs = []
for batch in range(n_batches):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)
    
x_umap = umap_op.fit_transform(np.concatenate(zs, axis = 0))

scores = pd.read_csv(result_dir + "score.csv", index_col = 0)
scores = scores[scores["methods"] == "scMoMaT"] 
resolution = scores["resolution"].values[np.argmax(scores["NMI (prec)"].values.squeeze())]
print(resolution)
# resolution = 1
resolution = 0.5
labels_tmp = utils.leiden_cluster(X = np.concatenate(zs, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
np.savetxt(result_dir + f'leiden_{K}_{T}_{resolution}.txt', labels_tmp)
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

utils.plot_latent_ext(x_umaps_scmomat, annos = labels, mode = "separate", save = result_dir + f'latent_separate_{K}_{T}.png', figsize = (10,20), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "x-large")

utils.plot_latent_ext(x_umaps_scmomat, annos = labels, mode = "modality", save = result_dir + f'latent_batches_{K}_{T}.png', figsize = (7,5), axis_label = "UMAP", markerscale = 10, s = 2, label_inplace = True, text_size = "x-large", alpha = 0.7)

utils.plot_latent_ext(x_umaps_scmomat, annos = labels, mode = "joint", save = result_dir + f'latent_clusters_{K}_{T}.png', figsize = (7,5), axis_label = "UMAP", markerscale = 10, s = 2, label_inplace = True, text_size = "xx-large", alpha = 0.7, colormap = "Paired")

utils.plot_latent_ext(x_umaps_scmomat, annos = prec_labels, mode = "separate", save = result_dir + f'latent_separate2_{K}_{T}.png', figsize = (10,20), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "x-large")

utils.plot_latent_ext(x_umaps_scmomat, annos = prec_labels, mode = "joint", save = result_dir + f'latent_clusters2_{K}_{T}.png', figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "x-large")

utils.plot_latent_ext(x_umaps_scmomat, annos = leiden_labels, mode = "joint", save = result_dir + f'latent_leiden_clusters_{K}_{T}_{resolution}.png', figsize = (7,5), axis_label = "UMAP", markerscale = 10, s = 2, label_inplace = True, text_size = "xx-large", alpha = 0.7)

# In[] Baseline methods
# UINMF
# uinmf_path = "pbmc/uinmf/"
uinmf_path = "pbmc/uinmf_bin/" 
H1 = pd.read_csv(uinmf_path + "H1_norm.csv", index_col = 0).values
H2 = pd.read_csv(uinmf_path + "H2_norm.csv", index_col = 0).values
H3 = pd.read_csv(uinmf_path + "H3_norm.csv", index_col = 0).values
H4 = pd.read_csv(uinmf_path + "H4_norm.csv", index_col = 0).values
liger_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(np.concatenate((H1, H2, H3, H4), axis = 0))
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

utils.plot_latent_ext(liger_umaps, annos = labels, mode = "separate", save = uinmf_path + f'latent_separate_uinmf.png', 
                      figsize = (10,20), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(liger_umaps, annos = prec_labels, mode = "separate", save = uinmf_path + f'latent_separate_uinmf_prec.png', 
                      figsize = (10,20), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(liger_umaps, annos = prec_labels, mode = "modality", save = uinmf_path + f'latent_batches_uinmf.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7)


# Multimap
multimap_path = "pbmc/multimap/"
batches = pd.read_csv(multimap_path + "batch_id.csv", index_col = 0)
X_multimap = np.load(multimap_path + "multimap.npy")
G_multimap = sp.load_npz(multimap_path + "multimap_graph.npz").todense()
X_multimaps = []
for batch in ["C1", "C2", "C3", "C4"]:
    X_multimaps.append(X_multimap[batches.values.squeeze() == batch, :])


utils.plot_latent_ext(X_multimaps, annos = labels, mode = "separate", save = multimap_path + f'latent_separate_multimap.png', 
                      figsize = (10,20), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(X_multimaps, annos = prec_labels, mode = "separate", save = multimap_path + f'latent_separate_multimap_prec.png', 
                      figsize = (10,20), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(X_multimaps, annos = prec_labels, mode = "modality", save = multimap_path + f'latent_batches_multimap.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7)


# In[]

import importlib 
importlib.reload(utils)
importlib.reload(bmk)
n_neighbors = 30
# graph connectivity score (gc) measure the batch effect removal per cell identity
# 1. scJMT
# construct neighborhood graph from the post-processed latent space
# knn_graph = np.zeros((knn_indices.shape[0], knn_indices.shape[0]))
# knn_graph[np.arange(knn_indices.shape[0])[:, None], knn_indices] = 1
# gc_scjmt = bmk.graph_connectivity(G = knn_graph, groups = np.concatenate(labels, axis = 0))

# 1. scMoMaT
gc_scmomat = bmk.graph_connectivity(X = np.concatenate(zs, axis = 0), groups = np.concatenate(prec_labels, axis = 0), k = n_neighbors)
print('GC (scMoMaT): {:.3f}'.format(gc_scmomat))

# 2. UINMF
gc_uinmf = bmk.graph_connectivity(X = np.concatenate((H1, H2, H3, H4), axis = 0), groups = np.concatenate(prec_labels, axis = 0), k = n_neighbors)
print('GC (UINMF): {:.3f}'.format(gc_uinmf))

# 3. Multimap
G_multimap[G_multimap == 0] = np.inf
knn_indices_multimap = G_multimap.argsort(axis = 1)[:, :n_neighbors]
knn_graph_multimap = np.zeros_like(G_multimap)
knn_graph_multimap[np.arange(knn_indices_multimap.shape[0])[:, None], knn_indices_multimap] = 1
gc_multimap = bmk.graph_connectivity(G = knn_graph_multimap, groups = np.concatenate(prec_labels, axis = 0), k = n_neighbors)
gc_multimap2 = bmk.graph_connectivity(X = np.concatenate(X_multimaps, axis = 0), groups = np.concatenate(prec_labels, axis = 0), k = n_neighbors)
print('GC (MultiMap): {:.3f}'.format(gc_multimap))
print('GC (MultiMap Graph): {:.3f}'.format(gc_multimap2))
# Batch effect removal regardless of cell identity
# Graph iLISI

# Conservation of biological identity
# NMI and ARI
# 1. scMoMaT
nmi_scmomat = []
ari_scmomat = []
for resolution in np.arange(0.1, 10, 0.5):
    # leiden_labels_scjmt = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
    leiden_labels_smomat = utils.leiden_cluster(X = np.concatenate(zs, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_scmomat.append(bmk.nmi(group1 = np.concatenate(prec_labels), group2 = leiden_labels_smomat))
    ari_scmomat.append(bmk.ari(group1 = np.concatenate(prec_labels), group2 = leiden_labels_smomat))
print('NMI (scMoMaT): {:.3f}'.format(max(nmi_scmomat)))
print('ARI (scMoMaT): {:.3f}'.format(max(ari_scmomat)))

# 2. UINMF
nmi_uinmf = []
ari_uinmf = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_uinmf = utils.leiden_cluster(X = np.concatenate((H1, H2, H3, H4), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_uinmf.append(bmk.nmi(group1 = np.concatenate(prec_labels), group2 = leiden_labels_uinmf))
    ari_uinmf.append(bmk.ari(group1 = np.concatenate(prec_labels), group2 = leiden_labels_uinmf))
print('NMI (UINMF): {:.3f}'.format(max(nmi_uinmf)))
print('ARI (UINMF): {:.3f}'.format(max(ari_uinmf)))

# 3. Multimap
G_multimap = sp.load_npz(multimap_path + "multimap_graph.npz").todense()
nmi_multimap = []
ari_multimap = []
for resolution in np.arange(0.1, 10, 0.5):
    # leiden_labels_seurat = utils.leiden_cluster(X = np.concatenate(seurat_pcas, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    # Multimap state to use graph for clustering
    leiden_labels_multimap = utils.leiden_cluster(affin = G_multimap, resolution = resolution)
    nmi_multimap.append(bmk.nmi(group1 = np.concatenate(prec_labels), group2 = leiden_labels_multimap))
    ari_multimap.append(bmk.ari(group1 = np.concatenate(prec_labels), group2 = leiden_labels_multimap))
print('NMI (MultiMap): {:.3f}'.format(max(nmi_multimap)))
print('ARI (MultiMap): {:.3f}'.format(max(ari_multimap)))

scores = pd.DataFrame(columns = ["methods", "resolution", "NMI", "ARI", "GC"])
scores["NMI (prec)"] = np.array(nmi_scmomat + nmi_uinmf + nmi_multimap)
scores["ARI (prec)"] = np.array(ari_scmomat + ari_uinmf + ari_multimap)
scores["GC (prec)"] = np.array([gc_scmomat] * len(nmi_scmomat) + [gc_uinmf] * len(nmi_uinmf) + [gc_multimap] * len(ari_multimap))
scores["resolution"] = np.array([x for x in np.arange(0.1, 10, 0.5)] * 3)
scores["methods"] = np.array(["scMoMaT"] * len(nmi_scmomat) + ["UINMF"] * len(nmi_uinmf) + ["MultiMap"] * len(ari_multimap))

# 1. scMoMaT
gc_scmomat = bmk.graph_connectivity(X = np.concatenate(zs, axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (scMoMaT): {:.3f}'.format(gc_scmomat))

# 2. UINMF
gc_uinmf = bmk.graph_connectivity(X = np.concatenate((H1, H2, H3, H4), axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (UINMF): {:.3f}'.format(gc_uinmf))

# 3. Multimap
G_multimap[G_multimap == 0] = np.inf
knn_indices_multimap = G_multimap.argsort(axis = 1)[:, :n_neighbors]
knn_graph_multimap = np.zeros_like(G_multimap)
knn_graph_multimap[np.arange(knn_indices_multimap.shape[0])[:, None], knn_indices_multimap] = 1
gc_multimap = bmk.graph_connectivity(G = knn_graph_multimap, groups = np.concatenate(labels, axis = 0), k = n_neighbors)
gc_multimap2 = bmk.graph_connectivity(X = np.concatenate(X_multimaps, axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (MultiMap): {:.3f}'.format(gc_multimap))
print('GC (MultiMap Graph): {:.3f}'.format(gc_multimap2))
# Batch effect removal regardless of cell identity
# Graph iLISI

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
    leiden_labels_uinmf = utils.leiden_cluster(X = np.concatenate((H1, H2, H3, H4), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_uinmf.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_uinmf))
    ari_uinmf.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_uinmf))
print('NMI (UINMF): {:.3f}'.format(max(nmi_uinmf)))
print('ARI (UINMF): {:.3f}'.format(max(ari_uinmf)))

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

scores["NMI"] = np.array(nmi_scmomat + nmi_uinmf + nmi_multimap)
scores["ARI"] = np.array(ari_scmomat + ari_uinmf + ari_multimap)
scores["GC"] = np.array([gc_scmomat] * len(nmi_scmomat) + [gc_uinmf] * len(nmi_uinmf) + [gc_multimap] * len(ari_multimap))
# scores["resolution"] = np.array([x for x in np.arange(0.1, 10, 0.5)] * 4)
# scores["methods"] = np.array(["scMoMaT"] * len(nmi_scmomat) + ["UINMF"] * len(nmi_uinmf) + ["MultiMap"] * len(ari_multimap))

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
ax.set_ylim(0, 0.7)
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
ax.set_ylim(0, 0.7)
# _ = ax.set_xlabel("cluster", fontsize = 20)
_ = ax.set_ylabel("ARI", fontsize = 20)
show_values_on_bars(ax)
fig.savefig(result_dir + "ARI.png", bbox_inches = "tight")    


# In[] Extend Motif
# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# 
# Extend scJMT to include the Motif obtained from chromVAR
# 
# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- # 
# read in dataset
dir = "../data/real/ASAP-PBMC/"
result_dir = "pbmc/scmomat/"
seurat_path = "pbmc/seurat/"
liger_path = "pbmc/liger/"

counts_rnas = []
counts_atacs = []
counts_motifs = []
counts_proteins = []
labels = []
n_batches = 4
for batch in range(n_batches):
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
        counts_protein = np.array(sp.load_npz(os.path.join(dir, 'PxC' + str(batch + 1) + ".npz")).todense().T)
        counts_protein = utils.preprocess(counts_protein, modality = "RNA", log = True)
    except:
        counts_protein = None

    try:
        counts_motif = pd.read_csv(dir + f'MxC{batch + 1}.csv', index_col = 0).T
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
    counts_proteins.append(counts_protein)
counts = {"rna":counts_rnas, "atac": counts_atacs, "protein": counts_proteins, "motif": counts_motifs}


A1 = sp.load_npz(os.path.join(dir, 'GxP.npz'))
A2 = sp.load_npz(os.path.join(dir, 'GxR.npz'))
A1 = np.array(A1.todense())
A2 = np.array(A2.todense())
# A1 = utils.preprocess(A1, modality = "interaction")
# A2 = utils.preprocess(A2, modality = "interaction")
# No need for pseudo-count matrix
# interacts = {"rna_atac": A2, "rna_protein": A1}

# obtain the feature name
genes = pd.read_csv(dir + "genes.txt", header = None).values.squeeze()
regions = pd.read_csv(dir + "regions.txt", header = None).values.squeeze()
proteins = pd.read_csv(dir + "proteins.txt", header = None).values.squeeze()

feats_name = {"rna": genes, "atac": regions, "protein": proteins, "motif": motifs}
counts["feats_name"] = feats_name

counts["nbatches"] = n_batches

# In[] retrain model, you can incorporate new matrices 
import importlib
importlib.reload(model)
# the leiden label is the one produced by the best resolution
alpha = [1000, 10]

model2 = model.cfrm_retrain_vanilla(model = model1, counts =  counts, labels = leiden_labels, alpha = alpha, device = device).to(device)
losses = model2.train(T = 4000)

x = np.linspace(0, 4000, int(4000/interval) + 1)
plt.plot(x, losses)

C_feats = {}
for mod in model2.mods:
    C_feat = model2.softmax(model2.C_feats[mod]).data.cpu().numpy() @ model2.A_assos["shared"].data.cpu().numpy().T 
    C_feats[mod] = pd.DataFrame(data = C_feat, index = model2.feats_name[mod], columns = ["cluster_" + str(i) for i in range(C_feat.shape[1])])

# In[]
C_gene = C_feats["rna"]
utils.plot_feat_score(C_gene, n_feats = 20, figsize= (15,20), save_as = result_dir + "C_gene.pdf", title = None)

C_protein = C_feats["protein"]
utils.plot_feat_score(C_protein, n_feats = 20, figsize = (17,20), save_as= result_dir + "C_protein.pdf", title = None)

C_motif = C_feats["motif"]
utils.plot_feat_score(C_motif, n_feats = 20, figsize= (20,20), save_as = result_dir + "C_motif.pdf", title = None)

C_region = C_feats["atac"]

C_gene.to_csv(result_dir + "C_gene.csv")
C_motif.to_csv(result_dir + "C_motif.csv")
C_region.to_csv(result_dir + "C_region.csv")
C_protein.to_csv(result_dir + "C_protein.csv")

C_gene = pd.read_csv(result_dir + "C_gene.csv", index_col = 0)
C_motif = pd.read_csv(result_dir + "C_motif.csv", index_col = 0)
C_region = pd.read_csv(result_dir + "C_region.csv", index_col = 0)
C_protein = pd.read_csv(result_dir + "C_protein.csv", index_col = 0)



# In[] Marker gene
# NOTE: Activation of naive T cells through the antigen-specific T cell receptor (TCR) initiates transcriptional programs that drive differentiation of lineage-specific effector functions; 
# 1. CD4+ T cells secrete cytokines to recruit and activate other immune cells
# 2. while CD8+ T cells acquire cytotoxic functions to directly kill infected or tumor cells. (Cytotxic CD8+ T cells, CD8_CTL?)
# 3. Most of these effector cells are short-lived, although some develop into long-lived memory T cells 
#       which persist as circulating central (TCM) 
#       and effector-memory (TEM) subsets, 
#       and non-circulating tissue resident memory T cells (TRM) in diverse lymphoid and non-lymphoid sites.

# CD4+ cells: 1. resting cells expressing CCR7, SELL and TCF7, (corresponding to naive or TCM cells), 
#             2. three activation-associated clusters expressing IL2, TNF, and IL4R at different levels
#             3. TRM-like resting and activated clusters expressing canonical TRM markers CXCR6 and ITGA1
#             4. distinct regulatory T cell (Treg) cluster expressing Treg- defining genes FOXP3, IL2RA, and CTLA4

# CD8+ cells: 1. two TEM/TRM-like clusters expressing CCL5, cytotoxicity- associated genes (GZMB, GZMK), and TRM markers (CXCR6, ITGA1);
#             2. activated TRM/TEM cluster expressing IFNG, CCL4, CCL3
#             3. clusters representing terminally differentiated effector cells (TEMRA) expressing cytotoxic markers PRF1 and NKG7


# factor 3 (NK): GNLY, NKG7, KLRB1, KLRD1, KLRF1

# factor 6 (Myeloid, no subclass): S100A9, LYZ (CD14+ Monocyte), NRD1 (Plasmaytoid dendritic cells), CD68 (Macrophage)
# factor 8 should be the same as factor 6

# factor 4 (B cells, no subclass): CD79A, CD37



# factor2 (Cytotxic CD8+ T cells): CD8B, CD8A, CD8B2, CD27 (T cells)
# factor 5 (Cytotxic CD8+ T cells?): GZMK (Cytotxic CD8+ T cells), KLRD1 (NK), CD8A (Cytotxic CD8+ T cells)
# factor 1 (Regulatory CD4+ T cells): FOXP3 (CCR4 not CCR10)
# factor 0, the same as factor 1
# factor 7. IL2 (IL2RA) (Regulatory CD4+ T cells)


# only the first two batches has counts rna
# factor 0, 1, 7 (from score below)
CD3E = [counts["rna"][0][:, counts["feats_name"]["rna"] == "CD3E"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "CD3E"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CD3E, mode = "joint", save = result_dir + "CD3E.png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "PuBu", title = "CD3E")
fig = utils.plot_factor(C_gene, markers = ["CD3E"], cluster = [0,1,2,5,7], figsize = (5,4))
fig.savefig(result_dir + "CD3E_score.png", bbox_inches = "tight")
CD3G = [counts["rna"][0][:, counts["feats_name"]["rna"] == "CD3G"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "CD3G"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CD3G, mode = "joint", save = result_dir + "CD3E.png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "PuBu", title = "CD3G")
fig = utils.plot_factor(C_gene, markers = ["CD3G"], cluster = [0,1,2,5,7], figsize = (5,4))
fig.savefig(result_dir + "CD3G_score.png", bbox_inches = "tight")
CD3D = [counts["rna"][0][:, counts["feats_name"]["rna"] == "CD3D"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "CD3D"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CD3D, mode = "joint", save = result_dir + "CD3D.png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "PuBu", title = "CD3D")
fig = utils.plot_factor(C_gene, markers = ["CD3D"], cluster = [0,1,2,5,7], figsize = (5,4))
fig.savefig(result_dir + "CD3D_score.png", bbox_inches = "tight")

CD4 = [counts["rna"][0][:, counts["feats_name"]["rna"] == "CD4"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "CD4"].squeeze() ]
# utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CD4, mode = "separate", save = result_dir + "CD4.png", figsize = (10, 15), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CD4, mode = "joint", save = result_dir + "CD4.png", figsize = (7, 5), axis_label = "UMAP", alpha = 1, cmap = "PuBu", title = "CD4")
fig = utils.plot_factor(C_gene, markers = ["CD4"], cluster = [0,1,7], figsize = (5,4))
fig.savefig(result_dir + "CD4_score.png", bbox_inches = "tight")

# factor 2, 5
CD8A = [counts["rna"][0][:, counts["feats_name"]["rna"] == "CD8A"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "CD8A"].squeeze() ]
CD8B = [counts["rna"][0][:, counts["feats_name"]["rna"] == "CD8B"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "CD8B"].squeeze() ]
# utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CD8A, mode = "separate", save = result_dir + "CD8A.png", figsize = (10, 15), axis_label = "UMAP", alpha = 0.5, cmap = "PuBu", title = "CD8A")
# utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CD8B, mode = "separate", save = result_dir + "CD8B.png", figsize = (10, 15), axis_label = "UMAP", alpha = 0.5, cmap = "PuBu", title = "CD8B")
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CD8A, mode = "joint", save = result_dir + "CD8A.png", figsize = (7, 5), axis_label = "UMAP", alpha = 0.5, cmap = "PuBu", title = "CD8A")
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CD8B, mode = "joint", save = result_dir + "CD8B.png", figsize = (7, 5), axis_label = "UMAP", alpha = 0.5, cmap = "PuBu", title = "CD8B")

fig = utils.plot_factor(C_gene, markers = ["CD8A"], cluster = [2,5], figsize = (5,4))
fig.savefig(result_dir + "CD8A_score.png", bbox_inches = "tight")
fig = utils.plot_factor(C_gene, markers = ["CD8B"], cluster = [2,5], figsize = (5,4))
fig.savefig(result_dir + "CD8B_score.png", bbox_inches = "tight")


# Factor 0, 2: Marker Naive using Protein CD45RA RESTING(high), CD45RO activated (low)， along with marker genes: CCR7, CD62L, CD27 to differ from CD45RA+ effectory memory cells
fig = utils.plot_factor(C_protein.iloc[:, [0,1,2,5,7]], markers = ["CD45RA"], cluster =[0,2], figsize = (5,4))
fig.savefig(result_dir + "CD45RA_score.png", bbox_inches = "tight")
CD45RA = [counts["protein"][0][:, counts["feats_name"]["protein"] == "CD45RA"].squeeze(), counts["protein"][1][:, counts["feats_name"]["protein"] == "CD45RA"].squeeze(), counts["protein"][2][:, counts["feats_name"]["protein"] == "CD45RA"].squeeze(), counts["protein"][3][:, counts["feats_name"]["protein"] == "CD45RA"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat, annos = CD45RA, mode = "joint", save = result_dir + "CD45RA.png", title = "CD45RA", figsize = (7, 5), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")

fig = utils.plot_factor(C_protein.iloc[:, [0,1,2,5,7]], markers = ["CD45RO"], cluster = [1,5,7], figsize = (5,4))
fig.savefig(result_dir + "CD45RO_score.png", bbox_inches = "tight")
CD45RO = [counts["protein"][0][:, counts["feats_name"]["protein"] == "CD45RO"].squeeze(), counts["protein"][1][:, counts["feats_name"]["protein"] == "CD45RO"].squeeze(), counts["protein"][2][:, counts["feats_name"]["protein"] == "CD45RO"].squeeze(), counts["protein"][3][:, counts["feats_name"]["protein"] == "CD45RO"].squeeze()]
utils.plot_latent_continuous(x_umaps_scmomat, annos = CD45RO, mode = "joint", save = result_dir + "CD45RO.png", title = "CD45RO",figsize = (7, 5), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")

# naive CD4+ cells (TCM) factor 0: CCR7, SELL, TCF7 (Naive, TCM)
fig = utils.plot_factor(C_gene.iloc[:, [0,1,2,5,7]], markers = ["CCR7"], cluster = [0,2], figsize = (7,5))
fig.savefig(result_dir + "CCR7_score.png", bbox_inches = "tight")
CCR7 = [counts["rna"][0][:, counts["feats_name"]["rna"] == "CCR7"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "CCR7"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CCR7, mode = "joint", save = result_dir + "CCR7.png", figsize = (10, 7), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")

fig = utils.plot_factor(C_gene.iloc[:, [0,1,2,5,7]], markers = ["SELL"], cluster = [0,2], figsize = (7,5))
fig.savefig(result_dir + "SELL_score.png", bbox_inches = "tight")
SELL = [counts["rna"][0][:, counts["feats_name"]["rna"] == "SELL"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "SELL"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = SELL, mode = "joint", save = result_dir + "SELL.png", figsize = (10, 7), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")

fig = utils.plot_factor(C_gene.iloc[:, [0,1,2,5,7]], markers = ["TCF7"], cluster = [0,2], figsize = (7,5))
fig.savefig(result_dir + "TCF7_score.png", bbox_inches = "tight")
TCF7 = [counts["rna"][0][:, counts["feats_name"]["rna"] == "TCF7"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "TCF7"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = TCF7, mode = "joint", save = result_dir + "TCF7.png", figsize = (10, 7), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")

# CD27 gene
fig = utils.plot_factor(C_gene.iloc[:, [0,1,2,5,7]], markers = ["CD27"], cluster = [0,2], figsize = (7,5))
fig.savefig(result_dir + "CD27_score.png", bbox_inches = "tight")
CD27 = [counts["rna"][0][:, counts["feats_name"]["rna"] == "CD27"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "CD27"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CD27, mode = "joint", save = result_dir + "CD27.png", figsize = (10, 7), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")
# CD27 protein
fig = utils.plot_factor(C_protein.iloc[:, [0,1,2,5,7]], markers = ["CD27"], cluster = [0,2], figsize = (7,5))
fig.savefig(result_dir + "CD27_protein_score.png", bbox_inches = "tight")
CD27 = [counts["protein"][0][:, counts["feats_name"]["protein"] == "CD27"].squeeze(), counts["protein"][1][:, counts["feats_name"]["protein"] == "CD27"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CD27, mode = "joint", save = result_dir + "CD27_protein.png", figsize = (10, 7), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")


# factor 1: Regulatory CD4+ T cells (Treg), factor 0 and 7 has low FOXP3
fig = utils.plot_factor(C_gene.iloc[:, [0,1,2,5,7]], markers = ["FOXP3"], cluster = [1], figsize = (7,5))
fig.savefig(result_dir + "FOXP3_score.png", bbox_inches = "tight")
FOXP3 = [counts["rna"][0][:, counts["feats_name"]["rna"] == "FOXP3"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "FOXP3"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = FOXP3, mode = "joint", save = result_dir + "FOXP3.png", figsize = (10, 7), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")
fig = utils.plot_factor(C_gene.iloc[:, [0,1,2,5,7]], markers = ["IL2RA"], cluster = [1], figsize = (7,5))
fig.savefig(result_dir + "IL2RA_score.png", bbox_inches = "tight")
IL2RA = [counts["rna"][0][:, counts["feats_name"]["rna"] == "IL2RA"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "IL2RA"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = IL2RA, mode = "joint", save = result_dir + "IL2RA.png", figsize = (10, 7), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")
fig = utils.plot_factor(C_gene.iloc[:, [0,1,2,5,7]], markers = ["CTLA4"], cluster = [1], figsize = (7,5))
fig.savefig(result_dir + "CTLA4_score.png", bbox_inches = "tight")
CTLA4 = [counts["rna"][0][:, counts["feats_name"]["rna"] == "CTLA4"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "CTLA4"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CTLA4, mode = "joint", save = result_dir + "CTLA4.png", figsize = (10, 7), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")





# T effector cells: IL2, TNF, and IL4R at different levels
# Factor 7: IL2, CD4+ effector (Th1), CD8+ effector (Tc1)
fig = utils.plot_factor(C_gene.iloc[:, [0,1,2,5,7]], markers = ["IL2"], cluster = [7], figsize = (7,5))
fig.savefig(result_dir + "IL2_score.png", bbox_inches = "tight")
IL2 = [counts["rna"][0][:, counts["feats_name"]["rna"] == "IL2"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "IL2"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = IL2, mode = "joint", save = result_dir + "IL2.png", figsize = (10, 7), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")

# TNF: CD4+ effector (Th1), CD8+ effector (Tc1) 
fig = utils.plot_factor(C_gene.iloc[:, [0,1,2,5,7]], markers = ["TNF"], cluster = [1,7], figsize = (7,5))
fig.savefig(result_dir + "TNF_score.png", bbox_inches = "tight")
TNF = [counts["rna"][0][:, counts["feats_name"]["rna"] == "TNF"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "TNF"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = TNF, mode = "joint", save = result_dir + "TNF.png", figsize = (10, 7), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")

# # Factor 8: CD4+ effector (Th1), TRM cell (ITGA1, ITGAE)
# fig = utils.plot_factor(C_gene.iloc[:, [0,1,2,5,7,8]], markers = ["ITGA1"], cluster = [8], figsize = (7,5))
# fig.savefig(result_dir + "ITGA1_score.png", bbox_inches = "tight")
# ITGA1 = [counts["rna"][0][:, counts["feats_name"]["rna"] == "ITGA1"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "ITGA1"].squeeze() ]
# utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = ITGA1, mode = "joint", save = result_dir + "ITGA1.png", figsize = (10, 7), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")

# fig = utils.plot_factor(C_gene.iloc[:, [0,1,2,5,7,8]], markers = ["ITGAE"], cluster = [8], figsize = (7,5))
# fig.savefig(result_dir + "ITGAE_score.png", bbox_inches = "tight")
# ITGAE = [counts["rna"][0][:, counts["feats_name"]["rna"] == "ITGAE"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "ITGAE"].squeeze() ]
# utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = ITGAE, mode = "joint", save = result_dir + "ITGAE.png", figsize = (10, 7), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")


# # IL4: Th2, Tc2
# _ = utils.plot_factor(C_gene.iloc[:, [0,1,2,5,7,8]], markers = ["IL4R"], cluster = [7], figsize = (10,7))
# IL4R = [counts["rna"][0][:, counts["feats_name"]["rna"] == "IL4R"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "IL4R"].squeeze() ]
# utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = IL4R, mode = "separate", save = None, figsize = (10, 15), axis_label = "UMAP", alpha = 0.5, cmap = "Greys")

# TEM: GZMA, CCR5, TBX21, (HLA-DRA,HLA-DRB1,HLA-DRB5)

# TRM: ITGA1, CXCR6 (CD69, ITGAE, CTLA4)
fig = utils.plot_factor(C_gene.iloc[:, [0,1,2,5,7]], markers = ["CD69"], cluster = [7], figsize = (7,5))
fig.savefig(result_dir + "CD69_score.png", bbox_inches = "tight")
CD69 = [counts["rna"][0][:, counts["feats_name"]["rna"] == "CD69"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "CD69"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CD69, mode = "joint", save = result_dir + "CD69.png", figsize = (10, 7), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")

# factor 5: TRM marker ITGA1, CXCR6, Cytotoxic associated genes: GZMB, GZMK

fig = utils.plot_factor(C_gene.iloc[:, [0,1,2,5,7]], markers = ["CXCR6"], cluster = [5], figsize = (7,5))
fig.savefig(result_dir + "CXCR6_score.png", bbox_inches = "tight")
CXCR6 = [counts["rna"][0][:, counts["feats_name"]["rna"] == "CXCR6"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "CXCR6"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = CXCR6, mode = "joint", save = result_dir + "CXCR6.png", figsize = (10, 7), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")

# Cytotoxic associated genes: GZMB, GZMK
fig = utils.plot_factor(C_gene.iloc[:, [0,1,2,5,7]], markers = ["GZMB"], cluster = [5], figsize = (7,5))
fig.savefig(result_dir + "GZMB_score.png", bbox_inches = "tight")
GZMB = [counts["rna"][0][:, counts["feats_name"]["rna"] == "GZMB"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "GZMB"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = GZMB, mode = "joint", save = result_dir + "GZMB.png", figsize = (10, 7), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")

fig = utils.plot_factor(C_gene.iloc[:, [0,1,2,5,7]], markers = ["GZMK"], cluster = [5], figsize = (7,5))
fig.savefig(result_dir + "GZMK_score.png", bbox_inches = "tight")
GZMK = [counts["rna"][0][:, counts["feats_name"]["rna"] == "GZMK"].squeeze(), counts["rna"][1][:, counts["feats_name"]["rna"] == "GZMK"].squeeze() ]
utils.plot_latent_continuous(x_umaps_scmomat[:2], annos = GZMK, mode = "joint", save = result_dir + "GZMK.png", figsize = (10, 7), axis_label = "UMAP", alpha = 0.5, cmap = "Reds")



# Summary: 0: CD4+ Naive, 1: Treg, 2: CD8+ Naive, 3: NK, 4: B cell,  5: Cytotoxic CD8+ T cell (TRM/Teffector), 6: Monocyte, 7,8: CD4+ effector

# In[]
de_genes = pd.read_csv(result_dir + "de_6&7.txt", index_col = 0)
de_genes = de_genes[de_genes["p_val_adj"] < 0.5]
for gene in de_genes.index.values:
    if C_gene.loc[gene, "cluster_6"] < C_gene.loc[gene, "cluster_7"]:
        fig = utils.plot_factor(C_gene, markers = [gene], cluster = [6,7], figsize = (5,4))
# In[] 
import bmk_graph
np.random.seed(0)
C_feats = {}
for mod in model1.mods:
    C_feats[mod] = model1.softmax(model1.C_feats[mod]).data.cpu().numpy()

protein2gene = C_feats["protein"] @ C_feats["rna"].T
AUPRC = bmk_graph.compute_auc_abs(G_inf = protein2gene, G_true = A1.T)
Eprec = bmk_graph.compute_eprec_abs(G_inf = protein2gene, G_true = A1.T)
protein2gene_rand = np.random.rand(protein2gene.shape[0], protein2gene.shape[1])
AUPRC_rand = bmk_graph.compute_auc_abs(G_inf = protein2gene_rand, G_true = A1.T)
Eprec_rand = bmk_graph.compute_eprec_abs(G_inf = protein2gene_rand, G_true = A1.T)
AUPRC_ratio = AUPRC/(AUPRC_rand + 1e-12)
Eprec_ratio = Eprec/(Eprec_rand + 1e-12)

print("AUPRC: {:.4F}, Eprec: {:.4f}".format(AUPRC, Eprec))
print("AUPRC ratio: {:.4F}, Eprec ratio: {:.4f}".format(AUPRC_ratio, Eprec_ratio))


region2gene = C_feats["atac"] @ C_feats["rna"].T
AUPRC = bmk_graph.compute_auc_abs(G_inf = region2gene, G_true = A2.T)
Eprec = bmk_graph.compute_eprec_abs(G_inf = region2gene, G_true = A2.T)
region2gene_rand = np.random.rand(region2gene.shape[0], region2gene.shape[1])
AUPRC_rand = bmk_graph.compute_auc_abs(G_inf = region2gene_rand, G_true = A2.T)
Eprec_rand = bmk_graph.compute_eprec_abs(G_inf = region2gene_rand, G_true = A2.T)
AUPRC_ratio = AUPRC/(AUPRC_rand + 1e-12)
Eprec_ratio = Eprec/(Eprec_rand + 1e-12)

print("AUPRC: {:.4F}, Eprec: {:.4f}".format(AUPRC, Eprec))
print("AUPRC ratio: {:.4F}, Eprec ratio: {:.4f}".format(AUPRC_ratio, Eprec_ratio))

# In[]
proteins = counts["feats_name"]["protein"]
genes = counts["feats_name"]["rna"]
regions = counts["feats_name"]["atac"]

ordering = np.argsort(protein2gene, axis = 1)[:, ::-1]
num_matched_protein = 0
for id, protein in enumerate(proteins):
    print(protein)
    top_score_gene = genes[ordering[id,:20]]
    gt_gene = genes[A1.T[id,:].astype(np.bool)]
    if set(gt_gene).issubset(set(top_score_gene)):
        num_matched_protein += 1

num_matched_protein /= len(proteins)
num_matched_protein

# In[]
ordering = np.argsort(region2gene.T, axis = 1)[:, ::-1]
num_matched_region = 0
for id, gene in enumerate(genes):
    print(gene)
    top_score_region = regions[ordering[id,:200]]
    gt_region = regions[A2[id,:].astype(np.bool)]
    if set(gt_region).issubset(set(top_score_region)):
        num_matched_region += 1

num_matched_region /= len(regions)
num_matched_region

# In[]
from scipy.spatial.distance import cosine
import seaborn as sns
plt.rcParams["font.size"] = 20
C_gene_p = A2 @ C_feats['atac'] 
sim = 0
for idx in range(C_feats['rna'].shape[0]):
    sim += cosine(C_gene_p[idx,:], C_feats['rna'][idx, :])
sim /= C_gene_p.shape[0]
print("the similarity between region factor and gene factor: {:.4f}".format(sim))


C_protein_p = A1.T @ C_feats['rna'] 
sim = 0
for idx in range(C_feats['protein'].shape[0]):
    sim += cosine(C_protein_p[idx,:], C_feats['protein'][idx, :])
sim /= C_protein_p.shape[0]
print("the similarity between gene factor and protein factor: {:.4f}".format(sim))

sim1 = []
for idx in range(C_feats['rna'].shape[0]):
    sim1.append(cosine(C_gene_p[idx,:], C_feats['rna'][idx, :]))

sim2 = []
for idx in range(C_feats['protein'].shape[0]):
    sim2.append(cosine(C_protein_p[idx,:], C_feats['protein'][idx, :]))

sim_df = pd.DataFrame(data = np.array(sim1 + sim2)[:, None], columns = ["cosine"])
sim_df["relationship"] = ["region & gene"] * len(sim1) + ["gene & protein"] * len(sim2)

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
sns.boxplot(data = sim_df, y = "cosine", x = "relationship", ax = ax)

fig.savefig(result_dir + "consistency.png", bbox_inches = "tight")


# %%
