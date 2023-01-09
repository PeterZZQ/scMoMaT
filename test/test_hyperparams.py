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
import seaborn as sns

from multiprocessing import Pool, cpu_count

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
dataset = "pbmc"
if dataset == "simulated":
    dir = "../data/simulated/6b16c_test_4/unequal/"
    result_dir = "simulated/6b16c_test_4/hyperparams/"
    scmomat_dir = result_dir

    if not os.path.exists(scmomat_dir):
        os.makedirs(scmomat_dir)

    n_batches = 6
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

    print(np.unique(np.concatenate(labels), return_counts = True))
elif dataset == "pbmc":
    dir = "../data/real/ASAP-PBMC/"
    result_dir = "pbmc/hyperparams/"
    scmomat_dir = result_dir

    if not os.path.exists(scmomat_dir):
        os.makedirs(scmomat_dir)

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
            counts_atac = sp.load_npz(os.path.join(dir, 'RxC' + str(batch + 1) + ".npz")).toarray().T
            counts_atac = utils.preprocess(counts_atac, modality = "ATAC")
        except:
            counts_atac = None
            
        try:
            counts_rna = sp.load_npz(os.path.join(dir, 'GxC' + str(batch + 1) + ".npz")).toarray().T
            counts_rna = utils.preprocess(counts_rna, modality = "RNA", log = False)
        except:
            counts_rna = None
        
        try:
            # the log transform produce better results for the protein
            counts_protein = sp.load_npz(os.path.join(dir, 'PxC' + str(batch + 1) + ".npz")).toarray().T
            counts_protein = utils.preprocess(counts_protein, modality = "RNA", log = True)
        except:
            counts_protein = None
        
        # preprocess the count matrix
        counts_rnas.append(counts_rna)
        counts_atacs.append(counts_atac)
        counts_proteins.append(counts_protein)

    counts = {"rna":counts_rnas, "atac": counts_atacs, "protein": counts_proteins}

    A1 = sp.load_npz(os.path.join(dir, 'GxP.npz')).toarray()
    A2 = sp.load_npz(os.path.join(dir, 'GxR.npz')).toarray()

    # obtain the feature name
    genes = pd.read_csv(dir + "genes.txt", header = None).values.squeeze()
    regions = pd.read_csv(dir + "regions.txt", header = None).values.squeeze()
    proteins = pd.read_csv(dir + "proteins.txt", header = None).values.squeeze()

    feats_name = {"rna": genes, "atac": regions, "protein": proteins}
    counts["feats_name"] = feats_name

    counts["nbatches"] = n_batches

# In[]
# NOTE: Running scmomat
# weight on regularization term
lambs = [1e-4, 1e-3, 1e-2]
batchsize = 0.1
# running seed
seed = 0
# number of latent dimensions
Ks = [80]
interval = 1000
T = 4000
lr = 1e-2

for K in Ks:
    for lamb in lambs:
        start_time = time.time()
        model1 = model.scmomat(counts = counts, K = K, batch_size = batchsize, interval = interval, lr = lr, lamb = lamb, seed = seed, device = device)
        losses1 = model1.train_func(T = T)
        torch.save(model1, scmomat_dir + f'scmomat_{K}_{T}_{lamb}.pt')
        end_time = time.time()
        print("running time: {:.3f}".format(end_time - start_time))

# In[]
n_neighbors_list = [15, 30, 50]
rs = [0.7, 0.9, None]
Ks = [80]
lambs = [1e-4, 1e-3, 1e-2]
T = 4000

scores = pd.DataFrame(columns = ["K", "lamb", "n_neighbor", "r", "NMI", "ARI", "GC"])
scores = pd.read_csv(result_dir + "/hyperparams_scores.csv", index_col = 0)
for K in Ks:
    for lamb in lambs:
        for n_neighbors in n_neighbors_list:
            for r in rs:
                model1 = torch.load(scmomat_dir + f'scmomat_{K}_{T}_{lamb}.pt')

                zs = []
                for batch in range(n_batches):
                    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
                    zs.append(z)

                s_pair_dist, knn_indices, knn_dists = utils.post_process(zs, n_neighbors, njobs = 8, r = r)

                n_neighbors =  knn_indices.shape[1]
                # graph connectivity score (gc) measure the batch effect removal per cell identity
                # 1. scMoMaT
                knn_graph = np.zeros((knn_indices.shape[0], knn_indices.shape[0]))
                knn_graph[np.arange(knn_indices.shape[0])[:, None], knn_indices] = 1
                gc_scmomat = bmk.graph_connectivity(G = knn_graph, groups = np.concatenate(labels, axis = 0))
                print('GC (scmomat): {:.3f}'.format(gc_scmomat))


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
                ari_scmomat = max(ari_scmomat)
                nmi_scmomat = max(nmi_scmomat)
                print('NMI (scMoMaT): {:.3f}'.format(nmi_scmomat))
                print('ARI (scMoMaT): {:.3f}'.format(ari_scmomat))

                # # Label transfer accuracy
                # # randomly select a half of cells as query
                # np.random.seed(0)
                # query_cell = np.array([False] * knn_indices.shape[0])
                # query_cell[np.random.choice(np.arange(knn_indices.shape[0]), size = int(0.5 * knn_indices.shape[0]), replace = False)] = True
                # training_cell = (1 - query_cell).astype(np.bool)
                # query_label = np.concatenate(labels)[query_cell]
                # training_label = np.concatenate(labels)[training_cell]
                # # scmomat
                # knn_graph = np.zeros((knn_indices.shape[0], knn_indices.shape[0]))
                # knn_graph[np.arange(knn_indices.shape[0])[:, None], knn_indices] = 1
                # knn_graph = knn_graph[query_cell, :][:, training_cell]
                # lta_scmomat = bmk.transfer_accuracy(query_label = query_label, train_label = training_label, knn_graph = knn_graph)

                # print("Label transfer accuracy (scMoMaT): {:.3f}".format(lta_scmomat))

                score = pd.DataFrame(data = np.array([[K, lamb, n_neighbors, r, nmi_scmomat, ari_scmomat, gc_scmomat]]), columns = ["K", "lamb", "n_neighbor", "r", "NMI", "ARI", "GC"])
                scores = pd.concat([scores, score])

scores.to_csv(result_dir + "/hyperparams_scores.csv")

# In[] Summarize scores
plt.rcParams["font.size"] = 20
if dataset == "simulated":
    scores = pd.DataFrame(columns = ["K", "lamb", "n_neighbor", "r", "NMI", "ARI", "GC"])

    for seed in [1,2,3,4,5]:
        result_dir = f'simulated/6b16c_test_{seed}/hyperparams/'
        score = pd.read_csv(result_dir + "hyperparams_scores.csv", index_col = 0)
        scores = pd.concat([scores, score])
elif dataset == "pbmc":
    scores = pd.read_csv(result_dir + "/hyperparams_scores.csv", index_col= 0)

scores["r"] = np.where(pd.isna(scores["r"].values), 1.0, scores["r"].values)
scores.columns = ["d", "lamb", "k", "r", "NMI", "ARI", "GC"]

fig = plt.figure(figsize = (20, 5))
ax = fig.subplots(nrows = 1, ncols = 3)
ax[0] = sns.barplot(data = scores, x = "d", hue = "lamb", y = "NMI",  ax = ax[0], capsize = 0.1)
ax[1] = sns.barplot(data = scores, x = "d", hue = "lamb", y = "ARI",  ax = ax[1], capsize = 0.1)
ax[2] = sns.barplot(data = scores, x = "d", hue = "lamb", y = "GC",  ax = ax[2], capsize = 0.1)

handles, labels = ax[2].get_legend_handles_labels()

if dataset == "pbmc":
    sns.stripplot(data = scores, x = "d", hue = "lamb", y = "NMI", ax = ax[0], color = "black", dodge = True)    
    sns.stripplot(data = scores, x = "d", hue = "lamb", y = "ARI", ax = ax[1], color = "black", dodge = True)  
    sns.stripplot(data = scores, x = "d", hue = "lamb", y = "GC", ax = ax[2], color = "black", dodge = True)  

ax[0].get_legend().remove()
ax[1].get_legend().remove()
ax[2].get_legend().remove()
l = plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title = "$\lambda$", frameon = False)


plt.tight_layout()
if dataset == "simulated":
    fig.savefig("simulated/K_lamb.png", bbox_inches = "tight")
elif dataset == "pbmc":
    fig.savefig(result_dir + "/K_lamb.png", bbox_inches = "tight")

fig = plt.figure(figsize = (20, 5))
ax = fig.subplots(nrows = 1, ncols = 3)
ax[0] =sns.barplot(data = scores, x = "k", hue = "r", y = "NMI",  ax = ax[0], capsize = 0.1)
ax[1] =sns.barplot(data = scores, x = "k", hue = "r", y = "ARI",  ax = ax[1], capsize = 0.1)
ax[2] =sns.barplot(data = scores, x = "k", hue = "r", y = "GC",  ax = ax[2], capsize = 0.1)

handles, labels = ax[2].get_legend_handles_labels()

if dataset == "pbmc":
    sns.stripplot(data = scores, x = "k", hue = "r", y = "NMI", ax = ax[0], color = "black", dodge = True)    
    sns.stripplot(data = scores, x = "k", hue = "r", y = "ARI", ax = ax[1], color = "black", dodge = True)  
    sns.stripplot(data = scores, x = "k", hue = "r", y = "GC", ax = ax[2], color = "black", dodge = True)  
ax[0].get_legend().remove()
ax[1].get_legend().remove()
ax[2].get_legend().remove()
l = plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title = "r", frameon = False)

plt.tight_layout()
if dataset == "simulated":
    fig.savefig("simulated/neighbor_r.png", bbox_inches = "tight")
elif dataset == "pbmc":
    fig.savefig(result_dir + "/neighbor_r.png", bbox_inches = "tight")

# %%
