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

import model
import utils
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# In[]
dir = "../data/real/ASAP-PBMC/"
result_dir = "pbmc/cfrm_quantile/"
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

start_time = time.time()
# model1 = model.cfrm_vanilla(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
model1 = model.cfrm_vanilla(counts = counts, K = K, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run, device = device).to(device)
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

utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = result_dir + f'latent_separate_{K}_{T}.png', figsize = (10,15), axis_label = "UMAP")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = result_dir + f'latent_batches_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = result_dir + f'latent_clusters_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent_ext(x_umaps, annos = prec_labels, mode = "separate", save = result_dir + f'latent_separate2_{K}_{T}.png', figsize = (10,15), axis_label = "UMAP")

utils.plot_latent_ext(x_umaps, annos = prec_labels, mode = "joint", save = result_dir + f'latent_clusters2_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6)

# In[]
n_neighbors = 30

zs = []
for batch in range(n_batches):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)

s_pair_dist, knn_indices, knn_dists = utils.re_nn_distance(zs, n_neighbors)
# s_pair_dist, knn_indices, knn_dists = re_distance_nn(zs, n_neighbors)

scores = pd.read_csv(result_dir + "score.csv", index_col = 0)
scores = scores[scores["methods"] == "scJMT"] 
resolution = scores["resolution"].values[np.argmax(scores["NMI"].values.squeeze())]
print(resolution)

labels_tmp = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
umap_op = UMAP(n_components = 2, n_neighbors = n_neighbors, min_dist = 0.1, random_state = 0, 
                metric='precomputed', knn_dists=knn_dists, knn_indices=knn_indices)
x_umap = umap_op.fit_transform(s_pair_dist)

# scDART
zs2 = utils.match_embeds(zs, k = n_neighbors, reference = None, bandwidth = 40)
# x_umap = UMAP(n_components = 2, min_dist = 0.2, random_state = 0).fit_transform(np.concatenate(zs2, axis = 0))
# labels_tmp = utils.leiden_cluster(X = np.concatenate(zs2, axis = 0), knn_indices = None, knn_dists = None, resolution = 0.3)


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

utils.plot_latent_ext(x_umaps, annos = leiden_labels, mode = "joint", save = result_dir + f'latent_leiden_clusters_{K}_{T}_processed.png', 
                      figsize = (15,10), axis_label = "Latent", markerscale = 6)

# In[] Baseline methods
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
H1 = pd.read_csv(liger_path + "liger_c1_norm.csv", sep = ",", index_col = 0).values
H2 = pd.read_csv(liger_path + "liger_c2_norm.csv", sep = ",", index_col = 0).values
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

# In[]
importlib.reload(bmk)

# graph connectivity score (gc) measure the batch effect removal per cell identity
# 1. scJMT
# construct neighborhood graph from the post-processed latent space
knn_graph = np.zeros((knn_indices.shape[0], knn_indices.shape[0]))
knn_graph[np.arange(knn_indices.shape[0])[:, None], knn_indices] = 1
gc_scjmt = bmk.graph_connectivity(G = knn_graph, groups = np.concatenate(labels, axis = 0))
print('GC (scJMT): {:.3f}'.format(gc_scjmt))

# 2. Seurat, n_neighbors affect the overall acc, and should be the same as scJMT
n_neighbors = knn_indices.shape[1]
gc_seurat = bmk.graph_connectivity(X = np.concatenate(seurat_pcas, axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (Seurat): {:.3f}'.format(gc_seurat))

# 3. Liger
gc_liger = bmk.graph_connectivity(X = np.concatenate((H1, H2), axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (Liger): {:.3f}'.format(gc_liger))

# 4. scJMT embedding
gc_scjmt_embed = bmk.graph_connectivity(X = np.concatenate(zs2, axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (scJMT embed): {:.3f}'.format(gc_scjmt_embed))

# Batch effect removal regardless of cell identity
# Graph iLISI

# Conservation of biological identity
# NMI and ARI
# 1. scJMT
nmi_scjmt = []
ari_scjmt = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_scjmt = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
    nmi_scjmt.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_scjmt))
    ari_scjmt.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_scjmt))
print('NMI (scJMT): {:.3f}'.format(max(nmi_scjmt)))
print('ARI (scJMT): {:.3f}'.format(max(ari_scjmt)))

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
    leiden_labels_liger = utils.leiden_cluster(X = np.concatenate((H1, H2), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_liger.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_liger))
    ari_liger.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_liger))
print('NMI (Liger): {:.3f}'.format(max(nmi_liger)))
print('ARI (Liger): {:.3f}'.format(max(ari_liger)))

# 1. scJMT embedding
nmi_scjmt_embed = []
ari_scjmt_embed = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_scjmt_embed = utils.leiden_cluster(X = np.concatenate(zs2, axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_scjmt_embed.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_scjmt_embed))
    ari_scjmt_embed.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_scjmt_embed))
print('NMI (scJMT embed): {:.3f}'.format(max(nmi_scjmt_embed)))
print('ARI (scJMT embed): {:.3f}'.format(max(ari_scjmt_embed)))

scores = pd.DataFrame(columns = ["methods", "resolution", "NMI", "ARI", "GC"])
scores["NMI"] = np.array(nmi_scjmt + nmi_seurat + nmi_liger + nmi_scjmt_embed)
scores["ARI"] = np.array(ari_scjmt + ari_seurat + ari_liger + ari_scjmt_embed)
scores["GC"] = np.array([gc_scjmt] * len(nmi_scjmt) + [gc_seurat] * len(nmi_seurat) + [gc_liger] * len(nmi_liger) + [gc_scjmt_embed] * len(nmi_scjmt_embed))
scores["resolution"] = np.array([x for x in np.arange(0.1, 10, 0.5)] * 4)
scores["methods"] = np.array(["scJMT"] * len(nmi_scjmt) + ["Seurat"] * len(nmi_seurat) + ["Liger"] * len(nmi_liger) + ["scJMT (embed)"] * len(nmi_scjmt_embed))
scores.to_csv(result_dir + "score.csv")

scores = pd.read_csv(result_dir + "score.csv")

# In[] Extend Motif
# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# 
# Extend scJMT to include the Motif obtained from chromVAR
# 
# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- # 
# read in dataset
dir = '../data/real/diag/Xichen/'
result_dir = "spleen/cfrm_quantile/"
seurat_path = "spleen/seurat/"
liger_path = "spleen/liger/"

counts_rnas = []
counts_atacs = []
counts_motifs = []
labels = []
n_batches = 2
for batch in range(1, n_batches+1):
    labels.append(pd.read_csv(os.path.join(dir, 'meta_c' + str(batch) + '.csv'), index_col=0)["cell_type"].values.squeeze())
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
    
    try:
        counts_motif = pd.read_csv(dir + r'MxC{}.csv'.format(batch), index_col = 0).T
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

A = sp.load_npz(os.path.join(dir, 'GxR.npz'))
A = np.array(A.todense())
interacts = None

genes = pd.read_csv(dir + "genes.txt", header = None).values.squeeze()
regions = pd.read_csv(dir + "regions.txt", header = None).values.squeeze()

feats_name = {"rna": genes, "atac": regions, "motif": motifs}
counts["feats_name"] = feats_name

# CALCULATE PSEUDO-SCRNA-SEQ
counts["rna"][1] = counts["atac"][1] @ A.T
#BINARIZE, still is able to see the cluster pattern, much denser than scRNA-Seq (cluster pattern clearer)
counts["rna"][1] = (counts["rna"][1]!=0).astype(int)

# PLOT FUNCTION
# x_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(np.concatenate(counts["rna"], axis = 0))
# utils.plot_latent_ext([x_umap[:counts["rna"][0].shape[0], :], x_umap[counts["rna"][0].shape[0]:, :]], annos = labels, mode = "separate", save = None, figsize = (10,15), axis_label = "UMAP")
# utils.plot_latent_ext([x_umap[:counts["rna"][0].shape[0], :], x_umap[counts["rna"][0].shape[0]:, :]], annos = labels, mode = "modality", save = None, figsize = (10,7), axis_label = "UMAP")

counts["nbatches"] = n_batches


# In[] retrain model, you can incorporate new matrices 
import importlib
importlib.reload(model)
# the leiden label is the one produced by the best resolution
model2 = model.cfrm_retrain_vanilla(model = model1, counts =  counts, labels = leiden_labels, device = device).to(device)
losses = model2.train(T = 2000)

x = np.linspace(0, 2000, int(2000/interval) + 1)
plt.plot(x, losses)

C_feats = {}
for mod in model2.mods:
    C_feat = model2.softmax(model2.C_feats[mod]).data.cpu().numpy() @ model2.A_assos["shared"].data.cpu().numpy().T 
    C_feats[mod] = pd.DataFrame(data = C_feat, index = model2.feats_name[mod], columns = ["cluster_" + str(i) for i in range(C_feat.shape[1])])

# In[]
C_gene = C_feats["rna"]
utils.plot_feat_score(C_gene, n_feats = 20, figsize= (15,20), save_as = None, title = None)

C_motif = C_feats["motif"]
utils.plot_feat_score(C_motif, n_feats = 20, figsize= (20,20), save_as = None, title = None)

C_region = C_feats["atac"]

C_gene.to_csv(result_dir + "C_gene.csv")
C_motif.to_csv(result_dir + "C_motif.csv")
C_region.to_csv(result_dir + "C_region.csv")
# %%
