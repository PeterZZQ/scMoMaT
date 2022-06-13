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
dir = "../data/real/MOp_5batches/"
result_dir = "MOp_5batches/scmomat/"


n_batches = 5
counts_rnas = []
counts_atacs = []
labels_ori = []
labels_trans = []
labels_remap = []
for batch in range(n_batches):
    # # read in labels
    # if batch == 0:
    #     labels_trans.append(pd.read_csv(os.path.join(dir, 'meta_c' + str(batch + 1) + '.csv'), index_col=0)["transferred.id"].values.squeeze())
    # elif batch == 1:
    #     labels_trans.append(pd.read_csv(os.path.join(dir, 'meta_c' + str(batch + 1) + '.csv'), index_col=0)["transferred.id"].values.squeeze())
    # elif batch == 2:
    #     labels_trans.append(pd.read_csv(os.path.join(dir, 'meta_c' + str(batch + 1) + '.csv'), index_col=0)["transferred.id"].values.squeeze())

    # read in labels
    if batch == 0:
        # labels.append(pd.read_csv(os.path.join(dir, 'meta_c' + str(batch + 1) + '_seurat.csv'), index_col=0)["celltype"].values.squeeze())
        label = pd.read_csv(os.path.join(dir, 'meta_c' + str(batch + 1) + '.csv'), index_col=0)["Ident"].values.squeeze()
        labels_ori.append(label)
        # Ast = (astrocytes = RG)
        # E2Rasgrf2 = Ex-L2/3-Rasgrf2
        # E3Rmst = Ex-L3/4-Rmst
        # E3Rorb = Ex-L3/4-Rorb
        # E4Il1rapl2 = Ex-L4/5-Il1rapl2
        # E4Thsd7a = Ex-L4/5-Thsd7a
        # E5Galnt14 = Ex-L5-Galnt14
        # E5Parm1 = Ex-L5-Parm1
        # E5Sulf1 = Ex-L5/6-Sulf1
        # E5Tshz2 = Ex-L5/6-Tshz2
        # E6Tle4 = Ex-L6-Tle4
        # Claustrum = Clau
        # migrating inhibitory neurons (In), based on marker genes Npy, Pvalb, Sst, Vip
        # InN = In-Npy
        # InP = In-Pvalb
        # InS = In-Sst
        # InV = In-Vip
        # Mic = Microglia
        # Mis = miscellaneous (unknown)
        # OPC = oligodendrocyte progenitor cells
        # OliI = Oli-Itpr2 (Itpr2-expressing oligodendrocyte, new oligodendrocyte)
        # OliM = Oli-Mal (mature oligodendrocyte)
        # Peri (Pericytes)
        label[label == "E2Rasgrf2"] = "L2/3"
        label[label == "E3Rmst"] = "L2/3"
        label[label == "E3Rorb"] = "L2/3"
        label[label == "E4Il1rapl2"] = "L4"
        label[label == "E4Thsd7a"] = "L4"
        label[label == "E5Galnt14"] = "L5"
        label[label == "E5Parm1"] = "L5"
        label[label == "E5Sulf1"] = "L5"
        label[label == "E5Tshz2"] = "L5"
        label[label == "E6Tle4"] = "L6"
        # astrocytes
        label[label == "Ast"] = "Astro"
        # Oligodendrocytes
        label[label == "OliM"] = "Oligo"
        label[label == "OliI"] = "Oligo"
        label[label == "OPC"] = "OPC"
        # Vip
        label[label == "InV"] = "CGE"
        label[label == "InS"] = "Sst"
        label[label == "InP"] = "Pvalb"
        # ?
        label[label == "InN"] = "Npy"
        label[label == "L5"] = "L4/5"
        label[label == "L4"] = "L4/5"
        label[label == "Mic"] = "MGC"
        
        labels_remap.append(label)
        
    elif batch == 1:
        # The cell type abbreviation largely follows the data paper: 
        # astrocytes (Astro), 
        # caudal ganglionic eminence interneurons (CGE), 
        # endothelial cells (Endo), 
        # layer 2 to layer 6 (L2-6), 
        # intratelencephalic neurons (intratelencephalically projecting, IT), 
        # extratelencephalically projecting (ET) neuron = pyramidal tracts (PT) neuron, 
        # corticothalamic neurons (CT), 
        # L6b excitatory neurons (L6b), 
        # microglial cells (MGC), 
        # near-projecting excitatory neurons (NP), 
        # oligodendrocytes (Oligo, OGC), 
        # oligodendrocyte precursors (OPC), 
        # smooth muscle cells (SMC), 
        # medial ganglionic eminence interneurons subclasses based on marker genes (Sst, Pvalb).
        # pericyte (peri); 
        # perivascular macrophage (PVM);  
        # vascular leptomeningeal cell (VLMC).
        # 
        # GABAergic inhibitory neurons include CGE, Sst and Pvalb
        # We divided GABAergic neurons into five major subclasses based on marker genes: 
        # Lamp5, Sncg and Vip, which label cells derived from the caudal ganglionic eminence (CGE), 
        # and Sst and Pvalb, which label cells derived from the medial ganglionic eminence. 
        # 
        # intratelencephalic neurons (IT) + L5 PT + L6 CT + L6b + NP = glutamatergic excitatory neurons
        label = pd.read_csv(os.path.join(dir, 'meta_c' + str(batch + 1) + '.csv'), index_col=0)["subclass_label"].values.squeeze()
        labels_ori.append(label)
        label[label == "Lamp5"] = "CGE"
        label[label == "Vip"] = "CGE"
        label[label == "Sncg"] = "CGE"
        
        # ?
        label[label == "L2/3 IT"] = "L2/3"
        label[label == "L5 ET"] = "L5"
        label[label == "L5 IT"] = "L5"
        label[label == "L6 CT"] = "L6"
        label[label == "L6 IT"] = "L6"
        label[label == "L6b"] = "L6"
        label[label == "L5"] = "L4/5"
        label[label == "L4"] = "L4/5"
        label[label == "L5/6 NP"] = "NP"
        label[label == "Macrophage"] = "MGC"

        label[label == "OPC"] = "OPC"
        labels_remap.append(label)

    elif batch == 2:
        # used by colbolt
        label = pd.read_csv(os.path.join(dir, 'meta_c' + str(batch + 1) + '.csv'), index_col=0)["MajorCluster"].values.squeeze()
        labels_ori.append(label)
        # label = pd.read_csv(os.path.join(dir, 'meta_c' + str(batch + 1) + '.csv'), index_col=0)["SubCluster"].values.squeeze()
        label[label == "L5.IT.a"] = "L5 IT"
        label[label == "L5.IT.b"] = "L5 IT"
        label[label == "L4"] = "L4"
        label[label == "L6.CT"] = "L6 CT"
        label[label == "L6.IT"] = "L6 IT"
        label[label == "L23.a"] = "L2/3"
        label[label == "L23.b"] = "L2/3"
        label[label == "L23.c"] = "L2/3"
        label[label == "OGC"] = "Oligo"

        # ?
        label[label == "ASC"] = "Astro"
        label[label == "OPC"] = "OPC"
        label[label == "Pv"] = "Pvalb"
        label[label == "L6 CT"] = "L6"
        label[label == "L6 IT"] = "L6"
        label[label == "L5 IT"] = "L5"
        label[label == "L5.PT"] = "L5"
        label[label == "L5"] = "L4/5"
        label[label == "L4"] = "L4/5"
        
        labels_remap.append(label)
    
    else:
        label = pd.read_csv(os.path.join(dir, 'meta_c' + str(batch + 1) + '.csv'), index_col=0)["MajorCluster"].values.squeeze()
        labels_ori.append(label)       
        label[label == "L2/3 IT"] = "L2/3"
        label[(label == "L4")|(label == "L5 IT")|(label == "L5 PT")] = "L4/5"
        label[(label == "L6 CT")|(label == "L6 IT")|(label == "L6b")] = "L6"
        label[(label == "Macrophage")] = "MGC"
        label[label == "Lamp5"] = "CGE"
        label[label == "Vip"] = "CGE"
        label[label == "Sncg"] = "CGE"        
        
        labels_remap.append(label)
    
    try:
        counts_atac = np.array(sp.load_npz(os.path.join(dir, 'RxC' + str(batch + 1) + ".npz")).todense().T)
        # counts_atac = np.array(sp.load_npz(os.path.join(dir, 'BxC' + str(batch + 1) + ".npz")).todense().T)
        counts_atac = utils.preprocess(counts_atac, modality = "ATAC")
        print("read atac for batch" + str(batch + 1))
        # x_lsi = lsi(counts_atac)
        # x_umap = UMAP(n_components = 2, min_dist = 0.1, random_state = 0).fit_transform(x_lsi)
        # print("umap atac for batch" + str(batch + 1))
        # utils.plot_latent_ext([x_umap], annos = [labels_remap[-1]], mode = "joint", save = result_dir + f'RxC{batch+1}', figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True)
        # utils.plot_latent_ext([x_umap], annos = [labels_trans[-1]], mode = "joint", save = result_dir + f'RxC{batch+1}_trans.png', figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True)

    except:
        counts_atac = None
        
    try:
        counts_rna = np.array(sp.load_npz(os.path.join(dir, 'GxC' + str(batch + 1) + ".npz")).todense().T)
        print("read rna for batch" + str(batch + 1))
        counts_rna = utils.preprocess(counts_rna, modality = "RNA", log = False)
        # x_pca = PCA(n_components = 30).fit_transform(np.log1p(counts_rna))
        # x_umap = UMAP(n_components = 2, min_dist = 0.1, random_state = 0).fit_transform(x_pca)
        # print("umap rna for batch" + str(batch + 1))
        # utils.plot_latent_ext([x_umap], annos = [labels_remap[-1]], mode = "joint", save = result_dir + f'GxC{batch+1}', figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True)
        # utils.plot_latent_ext([x_umap], annos = [labels_trans[-1]], mode = "joint", save = result_dir + f'GxC{batch+1}_trans.png', figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True)

    except:
        counts_rna = None
    
    # preprocess the count matrix
    counts_rnas.append(counts_rna)
    counts_atacs.append(counts_atac)

counts = {"rna":counts_rnas, "atac": counts_atacs}

# No need for pseudo-count matrix
A = sp.load_npz(os.path.join(dir, 'GxR.npz'))
A = np.array(A.todense())

# CALCULATE PSEUDO-SCRNA-SEQ
counts["rna"][2] = counts["atac"][2] @ A.T
#BINARIZE, still is able to see the cluster pattern, much denser than scRNA-Seq (cluster pattern clearer)
counts["rna"][2] = (counts["rna"][2]!=0).astype(int)

# CALCULATE PSEUDO-SCRNA-SEQ
counts["rna"][4] = counts["atac"][4] @ A.T
#BINARIZE, still is able to see the cluster pattern, much denser than scRNA-Seq (cluster pattern clearer)
counts["rna"][4] = (counts["rna"][4]!=0).astype(int)

# obtain the feature name
genes = pd.read_csv(dir + "genes.txt", header = None).values.squeeze()
regions = pd.read_csv(dir + "regions.txt", header = None).values.squeeze()
# regions = pd.read_csv(dir + "bins.txt", header = None).values.squeeze()

feats_name = {"rna": genes, "atac": regions}
counts["feats_name"] = feats_name

counts["nbatches"] = n_batches


# In[]
alpha = [1000, 1, 5]
batchsize = 0.1
run = 0
K = 30
interval = 1000
T = 4000
lr = 1e-2

start_time = time.time()
# model1 = model.cfrm_vanilla(counts = counts, K = K, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run, device = device).to(device)
# losses1 = model1.train_func(T = T)
# end_time = time.time()
# print("running time: " + str(end_time - start_time))

# x = np.linspace(0, T, int(T/interval)+1)
# plt.plot(x, losses1)
# save the model
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

utils.plot_latent_ext(x_umaps, annos = labels_remap, mode = "separate", save = result_dir + f'latent_separate_{K}_{T}.png', figsize = (15,30), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True)
utils.plot_latent_ext(x_umaps, annos = labels_remap, mode = "modality", save = result_dir + f'latent_batches_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True)

# In[]
import importlib 
importlib.reload(utils)
n_neighbors = 50

zs = []
for batch in range(n_batches):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)

s_pair_dist, knn_indices, knn_dists = utils.post_process(zs, n_neighbors, njobs = 8, r = None)

# scores = pd.read_csv(result_dir + "score.csv", index_col = 0)
# scores = scores[scores["methods"] == "scJMT"] 
# resolution = scores["resolution"].values[np.argmax(scores["NMI"].values.squeeze())]
# print(resolution)
resolution = 0.9
labels_tmp = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
umap_op = umap_batch.UMAP(n_components = 2, n_neighbors = n_neighbors, min_dist = 0.20, random_state = 0, 
                metric='precomputed', knn_dists=knn_dists, knn_indices=knn_indices)
x_umap = umap_op.fit_transform(s_pair_dist)

# np.save(result_dir + f'leiden_{K}_{T}_{resolution}.npy', labels_tmp)
# labels_tmp = np.load(result_dir + f'leiden_{K}_{T}_{resolution}.npy')

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

utils.plot_latent_ext(x_umaps, annos = labels_remap, mode = "separate", save = result_dir + f'latent_separate_{K}_{T}_processed.png', 
                      figsize = (10,27), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(x_umaps, annos = labels_remap, mode = "separate", save = result_dir + f'latent_separate_{K}_{T}_processed_anno.png', 
                      figsize = (10,27), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(x_umaps, annos = labels_remap, mode = "modality", save = result_dir + f'latent_batches_{K}_{T}_processed.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, alpha = 0.7)

utils.plot_latent_ext(x_umaps, annos = leiden_labels, mode = "joint", save = result_dir + f'latent_leiden_clusters_{K}_{T}_{resolution}_processed.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7)



# In[] Extend Motif
# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# 
# Extend scJMT to include the Motif obtained from chromVAR
# 
# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- # 
# read in dataset
n_batches = 5
counts_rnas = []
counts_atacs = []
counts_motifs = []
for batch in range(n_batches):
    # read in labels
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
        counts_motif = pd.read_csv(dir + f'MxC{batch+1}_raw.csv', index_col = 0).T
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

# obtain the feature name
genes = pd.read_csv(dir + "genes.txt", header = None).values.squeeze()
regions = pd.read_csv(dir + "regions.txt", header = None).values.squeeze()

# No need for pseudo-count matrix
A1 = sp.load_npz(os.path.join(dir, 'GxR.npz'))
# mmwrite(os.path.join(dir, "GxR.mtx"), A)
A1 = np.array(A1.todense())

A2 = pd.read_csv(dir + "region2motif_raw.csv", index_col = 0)
A2 = A2.loc[regions,:].astype(int)
# # CALCULATE PSEUDO-SCRNA-SEQ
# counts["rna"][2] = counts["atac"][2] @ A1.T
# #BINARIZE, still is able to see the cluster pattern, much denser than scRNA-Seq (cluster pattern clearer)
# counts["rna"][2] = (counts["rna"][2]!=0).astype(int)

feats_name = {"rna": genes, "atac": regions, "motif": motifs}
counts["feats_name"] = feats_name

counts["nbatches"] = n_batches


# In[] retrain model, you can incorporate new matrices 
import importlib
importlib.reload(model)
# the leiden label is the one produced by the best resolution
alpha = [1000, 10]

model2 = model.cfrm_retrain_vanilla(model = model1, counts =  counts, labels = leiden_labels, alpha = alpha, device = device).to(device)
losses = model2.train(T = 2000)

x = np.linspace(0, 2000, int(2000/interval) + 1)
plt.plot(x, losses)

C_feats = {}
for mod in model2.mods:
    C_feat = model2.softmax(model2.C_feats[mod]).data.cpu().numpy() @ model2.A_assos["shared"].data.cpu().numpy().T 
    C_feats[mod] = pd.DataFrame(data = C_feat, index = model2.feats_name[mod], columns = ["cluster_" + str(i) for i in range(C_feat.shape[1])])

# In[]
C_gene = C_feats["rna"]
utils.plot_feat_score(C_gene, n_feats = 20, figsize= (15,35), save_as = result_dir + "C_gene.pdf", title = None)

C_motif = C_feats["motif"]
utils.plot_feat_score(C_motif, n_feats = 20, figsize= (20,35), save_as = result_dir + "C_motif.pdf", title = None)

C_region = C_feats["atac"]

C_gene.to_csv(result_dir + "C_gene.csv")
C_motif.to_csv(result_dir + "C_motif.csv")
C_region.to_csv(result_dir + "C_region.csv")

C_gene = pd.read_csv(result_dir + "C_gene.csv", index_col = 0)
C_motif = pd.read_csv(result_dir + "C_motif.csv", index_col = 0)
C_region = pd.read_csv(result_dir + "C_region.csv", index_col = 0)

# import seaborn as sns
# sns.clustermap(data = C_gene)


# In[] 
# Checked, Factor 0: L6 CT/L6 b, marker (up) Slc17a7, Fezf2, Sulf1, Foxp2 (unique compared to L6 IT). (down, unique compared to L6 IT) Slc30a3,  
# Checked, Factor 6: L6 IT marker (up) Slc17a7, Fezf2, Sulf1, Slc30a3, (down) Foxp2
sub_dir = result_dir + "L6CT_L6b/"
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)
genes = ['Slc17a7', 'Fezf2', 'Sulf1', 'Foxp2', 'Slc30a3']
fig = utils.plot_factor(C_gene, markers = genes, cluster = 0, figsize = (12, 10))
fig.savefig(sub_dir + "marker_genes.png")

sub_dir = result_dir + "L6IT/"
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)
fig = utils.plot_factor(C_gene, markers = genes, cluster = [10, 2], figsize = (12, 10))
fig.savefig(sub_dir + "marker_genes.png")

# In[] Checked, Factor 8 NP, L5/6 NP, Slc17a7, Fezf2, Sla2, Foxp2, Tshz2
# Tshz2 consistently marks L5 NP cell types across data modalities [A transcriptomic and epigenomic cell atlas of the mouse primary motor cortex]
sub_dir = result_dir + "NP/"
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)
genes = ['Fezf2', 'Slc17a7', 'Sla2', 'Foxp2', 'Tshz2']
fig = utils.plot_factor(C_gene, markers = genes, cluster = 7, figsize = (10,10))
fig.savefig(sub_dir + "marker_genes.png")

# In[] Checked, Factor 1 L2/3 'Slc17a7', 'Slc30a3', 'Rfx3', 'Rfx4', 'Lamp5', 'Calb1', 'Otof'
sub_dir = result_dir + "L23/"
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)
genes = ['Slc17a7', 'Slc30a3', 'Rfx3', 'Rfx4', 'Lamp5', 'Calb1']
fig = utils.plot_factor(C_gene, markers = genes, cluster = [1], figsize = (10,10))
fig.savefig(sub_dir + "marker_genes.png")

# In[] 
# Checked, Factor 2/4/5/7 L4/5 (L4) Rorb,Rspo1,Slc30a3,Slc17a7 (up), Foxp2(down)
# L4: Cux2, Rspo1 and Rorb(both clusters), L5: Fezf2 (one cluster)
# We provide new evidence that the MOp has an excitatory neuron population that expresses 
# markers of L4 thalamic-recipient neurons, including Cux2, Rspo1 and Rorb4. 
# up regulated
sub_dir = result_dir + "L45/"
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)
genes = ['Rorb', 'Rspo1', 'Cux2', 'Slc30a3', 'Slc17a7', 'Foxp2', 'Fezf2']
fig = utils.plot_factor(C_gene, markers = genes, cluster = [3], figsize = (10,12))
fig.savefig(sub_dir + "marker_genes.png")

# # Factor 7: L5 PT (ET), marker: Fezf2, Fam84b (not here), Bcl6[https://www.nature.com/articles/s41586-018-0654-5.pdf]. Other highly expressed: Slc17a7, unclear
# fig = utils.plot_factor(C_gene, markers = ['Fezf2', 'Bcl6'], cluster = 7, figsize = (10, 4))

# In[] Checked, Factor 10 Astro ["Slc1a2", "Aldoc", "Plpp3", "Slc1a3", "Sparcl1", "Cst3", "Mt3", "Apoe", "Atp1a2", "Id3", "Fabp7", "Aqp4", "Glul", "Clu", "Mfge8", "Cpe", "Slc4a4", "Mt1", "Pla2g7", "Gja1"]
sub_dir = result_dir + "Astro/"
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)
genes = ["Slc1a2", "Aldoc", "Slc1a3", "Sparcl1", "Cst3", "Apoe", "Id3", "Fabp7", "Glul", "Clu", "Mfge8", "Slc4a4", "Mt1", "Pla2g7", "Gja1"]
fig = utils.plot_factor(C_gene, markers = genes, cluster = 8, figsize = (10,25))
fig.savefig(sub_dir + "marker_genes.png")


# In[] Factor 11, Macrophage, "C1qb","C1qa","Ccl4","C1qc","Hexb","Tyrobp","Ccl2","Fcer1g","Ctss","Ccl3","Csf1r","Lgmn","Cx3cr1","Pf4","P2ry12","Fcrls","Sepp1","Ctsd","Trem2","Laptm5"
sub_dir = result_dir + "Macrophage/"
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)
genes = ["C1qb","C1qa","Hexb","Fcer1g","Ctss","Csf1r","Lgmn","Cx3cr1","Trem2"]
fig = utils.plot_factor(C_gene, markers = genes, cluster = 12, figsize = (10,15))
fig.savefig(sub_dir + "marker_genes.png")

# In[] Factor 12, OPC: "Pdgfra", "Lhfpl3", "Olig1", "C1ql1", "S100a13", "Olig2", "Epn2", "Cspg4", "Scrg1", "Matn4", "Cntn1", "S100a1", "Pllp", "Cdo1", "Sox10", "Gpr17", "Cspg5", "Ostf1", "Tpm1", "Ncald"
sub_dir = result_dir + "OPC/"
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)
genes = ["Lhfpl3", "Olig1", "Scrg1", "Matn4", "S100a1", "Pllp", "Sox10", "Cspg5", "Ostf1"]
fig = utils.plot_factor(C_gene, markers = genes, cluster = 11, figsize = (10, 15))
fig.savefig(sub_dir + "marker_genes.png")

# In[] Factor 9, Oligo: Plp1, Mbp, Bcas1, Sirt2, Cnp, Mag, Cldn11, Fyn, Gpr17, Enpp6, Bmp4, Cd9, Tubb4a, Nfasc, Lims2, Tnr, Rnd2, Dynll2, Ugt8a, Mobp
sub_dir = result_dir + "Oligo/"
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)
genes = ["Plp1", "Mbp", "Bcas1", "Cnp", "Mag", "Cldn11", "Enpp6", "Cd9", "Tubb4a", "Lims2", "Ugt8a", "Mobp"]
fig = utils.plot_factor(C_gene, markers = genes, cluster = 9, figsize = (10, 20))
fig.savefig(sub_dir + "marker_genes.png")
# In[] Factor 3: Pvalb, Sst; factor 6, Lamp5, Sncg, Vip, Npy ()
sub_dir = result_dir + "GABAergic/"
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)
genes = ["Pvalb", "Sst", "Npy", "Lamp5", "Sncg", "Vip"]
fig = utils.plot_factor(C_gene, markers = genes, cluster = [4,6], figsize = (10, 12))
fig.savefig(sub_dir + "marker_genes.png")

# In[] Motifs
# L2/3 Rfx (enriched), Mads (slightly enriched), Tal, NK, Pou, Arid (deplete)
fig = utils.plot_factor(C_motif, markers = ["MA0509.1_Rfx1", "MA0627.1_Pou2f3", "MA0142.1_Pou5f1::Sox2", "MA0601.1_Arid3b", "MA0602.1_Arid5a", "MA0151.1_Arid3a", "MA0124.2_Nkx3-1", "MA0063.1_Nkx2-5", "MA0503.1_Nkx2-5(var.2)"], cluster = 1, figsize = (10,12))

# L6 CT Rfx (enriched), Mads (slightly enriched), Tal, NK, Pou, Arid (deplete)
fig = utils.plot_factor(C_motif, markers = ["MA0509.1_Rfx1", "MA0627.1_Pou2f3", "MA0142.1_Pou5f1::Sox2", "MA0601.1_Arid3b", "MA0602.1_Arid5a", "MA0151.1_Arid3a", "MA0124.2_Nkx3-1", "MA0063.1_Nkx2-5", "MA0503.1_Nkx2-5(var.2)"], cluster = 1, figsize = (10,12))

# Factor 10, Astro: Checked
# MA0135.1_Lhx3, MA0704.1_Lhx4, MA0705.1_Lhx8, 
# (NK family) MA0063.1_Nkx2-5, MA0503.1_Nkx2-5(var.2)
# (Hox family) MA0912.1_Hoxd3, MA0904.1_Hoxb5, MA0910.1_Hoxd8, MA0911.1_Hoxa11, MA0913.1_Hoxd9
# (Sox family) MA0514.1_Sox3, MA0442.1_SOX10

# Factor 11, Macrophage
# MA0062.2_Gabpa, MA0117.2_Mafb, MA0002.2_RUNX1 (MA0742.1_Klf12, MA0065.2_Pparg::Rxra)

# Factor 12 OPC
# MA0442.1_SOX10, MA0514.1_Sox3, MA0515.1_Sox6, MA0143.3_Sox2, MA0521.1_Tcf12

# Factor 9 Oligo
# MA0515.1_Sox6, MA0442.1_SOX10, MA0514.1_Sox3, MA0160.1_NR4A2 (NR4)

# Factor 3 GABAergic inhibitory neuron(Pvalb, Sst; factor 6, Lamp5, Sncg, Vip, Npy (mix, jump))

# Factor 1 L23: MA0509.1_Rfx1, MA0047.2_Foxa2

# Factor 8 NP: MA0623.1_Neurog1, MA0607.1_Bhlha15, MA0461.2_Atoh1, MA0633.1_Twist2, MA0463.1_Bcl6

# Factor 0: L6b: MA0463.1_Bcl6, MA0518.1_Stat4, MA0631.1_Six3(MA0607.1_Bhlha15, MA0482.1_Gata4)

# Factor 2/4/5/7 L4/5: MA0832.1_Tcf21, MA0623.1_Neurog1, MA0500.1_Myog, MA0499.1_Myod1 (L4)

# In[] Interactions
import bmk_graph
C_feats = {}
for mod in model2.mods:
    C_feats[mod] = model2.softmax(model2.C_feats[mod]).data.cpu().numpy()
# for mod in model1.mods:
#     C_feats[mod] = model1.softmax(model1.C_feats[mod]).data.cpu().numpy()

motif2gene_gt = ((A1 @ A2.values) > 0).astype(int)
motif2gene = C_feats['motif'] @ C_feats['rna'].T
np.random.seed(0)
motif2gene_rand = np.random.rand(motif2gene.shape[0], motif2gene.shape[1])
AUPRC = bmk_graph.compute_auc_abs(G_inf = motif2gene, G_true = motif2gene_gt)
Eprec = bmk_graph.compute_eprec_abs(G_inf = motif2gene, G_true = motif2gene_gt)
AUPRC_rand = bmk_graph.compute_auc_abs(G_inf = motif2gene_rand, G_true = motif2gene_gt)
Eprec_rand = bmk_graph.compute_eprec_abs(G_inf = motif2gene_rand, G_true = motif2gene_gt)
AUPRC_ratio = AUPRC/AUPRC_rand
Eprec_ratio = Eprec/Eprec_rand

print("AUPRC: {:.4F}, Eprec: {:.4f}".format(AUPRC, Eprec))
print("AUPRC ratio: {:.4F}, Eprec ratio: {:.4f}".format(AUPRC_ratio, Eprec_ratio))

# In[]
_ = plt.hist(motif2gene.reshape(-1), bins = 30)
_ = plt.hist(motif2gene_gt.reshape(-1), bins = 30)
# find the largest to small values
order = (-motif2gene).argsort(axis = None, kind='mergesort')
indices = np.vstack(np.unravel_index(order, motif2gene.shape)).T
pairs = []
for index in indices:
    motif = model2.feats_name["motif"][index[0]]
    gene = model2.feats_name["rna"][index[1]]
    pairs.append([motif, gene])

# In[] regions correspond to the gene body, use the gene activity matrix, very low
from scipy.spatial.distance import cosine
import seaborn as sns
C_gene_p = (A1 @ A2.values > 0).astype(int) @ C_feats['motif'] 
sim = 0
for idx in range(C_feats['rna'].shape[0]):
    sim += cosine(C_gene_p[idx,:], C_feats['rna'][idx, :])
sim /= C_gene.shape[0]
print("the similarity between motif factor and gene factor: {:.4f}".format(sim))

C_feats = {}
for mod in model1.mods:
    C_feats[mod] = model1.softmax(model1.C_feats[mod]).data.cpu().numpy()

C_gene_p = A1 @ C_feats['atac']
sim = 0
for idx in range(C_feats['rna'].shape[0]):
    sim += cosine(C_gene_p[idx,:], C_feats['rna'][idx, :])
sim /= C_feats['rna'].shape[0]
print("the similarity between region factor and gene factor: {:.4f}".format(sim))


plt.rcParams["font.size"] = 20
C_gene_p = A1 @ C_feats['atac']
sim = []
for idx in range(C_feats['rna'].shape[0]):
    sim.append(cosine(C_gene_p[idx,:], C_feats['rna'][idx, :]))
sim = pd.DataFrame(data = np.array(sim)[:, None], columns = ["cosine"])
fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
sns.boxplot(data = sim, x = "cosine", ax = ax)
ax.set_xlim([0,1])

fig.savefig(result_dir + "consistency.png", bbox_inches = "tight")



# %%
