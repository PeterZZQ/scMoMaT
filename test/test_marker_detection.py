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
import scipy.stats as stats

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
dir = "../data/simulated/de_test_5/"
result_dir = "simulated/de_test_5/scenario2"
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

if result_dir[-1] == "1":
    print("scenario1")
    # NOTE: SCENARIO 1: diagonal integration
    counts["rna"][0] = None
    counts["rna"][1] = None
    counts["rna"][2] = None
    counts["atac"][3] = None 
    counts["atac"][4] = None
    counts["atac"][5] = None
elif result_dir[-1] == "2":
    print("scenario2")
    # NOTE: SCENARIO 2: diagonal with partial shared
    counts["rna"][0] = None
    counts["rna"][1] = None
    counts["rna"][2] = None
    counts["atac"][4] = None
    counts["atac"][5] = None
else:
    assert False

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

de_analysis = pd.read_csv(os.path.join(dir, "de_genes.txt"), sep = "\t")
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

start_time = time.time()
model1 = model.scmomat_model(counts = counts, K = K, batch_size = batchsize, interval = interval, lr = lr, lamb = lamb, seed = seed, device = device)
losses1 = model1.train_func(T = T)
end_time = time.time()
print("running time: " + str(end_time - start_time))

torch.save(model1, scmomat_dir + f'CFRM_{K}_{T}.pt')
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

# In[] find the resolution with the largest NMI
nmi_scmomat = []
ari_scmomat = []
for resolution in np.arange(0.1, 10, 0.5):
    # use the post-processed graph
    leiden_labels_smomat = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
    nmi_scmomat.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_smomat))
    ari_scmomat.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_smomat))
print('NMI (scMoMaT): {:.3f}'.format(max(nmi_scmomat)))
print('ARI (scMoMaT): {:.3f}'.format(max(ari_scmomat)))
resolution = np.arange(0.1, 10, 0.5)[np.argmax(nmi_scmomat)]
print(f"the best resolution: {resolution}")
labels_tmp = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)

umap_op = umap_batch.UMAP(n_components = 2, n_neighbors = n_neighbors, min_dist = 0.30, random_state = 0, 
                metric='precomputed', knn_dists=knn_dists, knn_indices=knn_indices)
x_umap = umap_op.fit_transform(s_pair_dist)


# separate into batches
x_umaps = []
leiden_labels_scmomat = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
        leiden_labels_scmomat.append(labels_tmp[start_pointer:end_pointer])

    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        x_umaps.append(x_umap[start_pointer:,:])
        leiden_labels_scmomat.append(labels_tmp[start_pointer:])

    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        x_umaps.append(x_umap[start_pointer:end_pointer,:])
        leiden_labels_scmomat.append(labels_tmp[start_pointer:end_pointer])


utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = scmomat_dir + f'latent_separate_{K}_{T}_processed.png', 
                      figsize = (7,20), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = scmomat_dir + f'latent_joint_{K}_{T}_processed.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, alpha = 0.7, text_size = "x-large")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = scmomat_dir + f'latent_batches_{K}_{T}_processed.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, alpha = 0.7)

utils.plot_latent_ext(x_umaps, annos = leiden_labels_scmomat, mode = "joint", save = scmomat_dir + f'latent_leiden_clusters_{K}_{T}_{resolution}_processed.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, alpha = 0.7)

# In[] Retraining
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

if result_dir[-1] == "1":
    print("scenario1")
    # NOTE: SCENARIO 1: diagonal integration
    counts["rna"][0] = None
    counts["rna"][1] = None
    counts["rna"][2] = None
    counts["atac"][3] = None 
    counts["atac"][4] = None
    counts["atac"][5] = None
elif result_dir[-1] == "2":
    print("scenario2")
    # NOTE: SCENARIO 2: diagonal with partial shared
    counts["rna"][0] = None
    counts["rna"][1] = None
    counts["rna"][2] = None
    counts["atac"][4] = None
    counts["atac"][5] = None
else:
    assert False

# obtain the feature name
genes = np.array(["gene_" + str(x) for x in range(counts["rna"][-1].shape[1])])
regions = np.array(["region_" + str(x) for x in range(counts["atac"][0].shape[1])])
feats_name = {"rna": genes, "atac": regions}
counts["feats_name"] = feats_name
counts["nbatches"] = n_batches

de_analysis = pd.read_csv(os.path.join(dir, "de_genes.txt"), sep = "\t")


# retraining the model
lamb = 0.01

# the leiden label is the one produced by the best resolution
model2 = model.scmomat_retrain(model = model1, counts =  counts, labels = leiden_labels_scmomat, lamb = lamb, device = device)
losses = model2.train(T = 4000)

x = np.linspace(0, 4000, int(4000/interval) + 1)
plt.plot(x, losses)
plt.yscale("log")

C_feats = {}
for mod in model2.mods:
    C_feat = model2.softmax(model2.C_feats[mod]).data.cpu().numpy() @ model2.A_assos["shared"].data.cpu().numpy().T 
    C_feats[mod] = pd.DataFrame(data = C_feat, index = model2.feats_name[mod], columns = ["cluster_" + str(i) for i in range(C_feat.shape[1])])


score_genes_scmomat = C_feats["rna"]
score_genes_scmomat.to_csv(scmomat_dir + "score_genes_scmomat.csv")

# In[]

# calculate mapping of leiden cluster results and ground truth clusters
cluster_mapping_scmomat = {}
uniq_clusters = np.unique(np.concatenate(leiden_labels_scmomat))
for cluster in uniq_clusters:
    cluster_idx = np.where(np.concatenate(leiden_labels_scmomat) == cluster)[0]
    gt_clusters, cell_counts = np.unique(np.concatenate(labels)[cluster_idx], return_counts = True)
    cluster_mapping_scmomat[cluster] = gt_clusters[np.argsort(cell_counts)[-1]]


kt_scores_scmomat = []
for cluster in uniq_clusters:
    gt_cluster = cluster_mapping_scmomat[cluster]
    # rank based comparison
    # log-fold change
    logFC = de_analysis.loc[de_analysis["pop"] == gt_cluster, "logFC_theoretical"].values.squeeze()
    # extract the genes that are enriched in the population, log fold change is above 0
    gene_enrich = np.where(logFC > 0)[0]
    kt_score, _ = stats.kendalltau(score_genes_scmomat.iloc[gene_enrich, cluster].values.squeeze(), logFC[gene_enrich])
    print("with logFC: {:f}".format(kt_score))

    # p-value
    scores_gt = de_analysis.loc[de_analysis["pop"] == gt_cluster, "wil.p_true_counts"].values.squeeze()
    kt_score, _ = stats.kendalltau(- score_genes_scmomat.iloc[gene_enrich, cluster].values.squeeze(), scores_gt[gene_enrich])
    print("with p value: {:f}".format(kt_score))

    scores_gt = de_analysis.loc[de_analysis["pop"] == gt_cluster, "wil.p_theoretical"].values.squeeze()
    kt_score, _ = stats.kendalltau(- score_genes_scmomat.iloc[gene_enrich, cluster].values.squeeze(), scores_gt[gene_enrich])
    kt_scores_scmomat.append(kt_score)
    print("with p value: {:f}".format(kt_score))

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

nmi_uinmf = []
ari_uinmf = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_uinmf = utils.leiden_cluster(X = np.concatenate((H1_uinmf, H2_uinmf, H3_uinmf, H4_uinmf, H5_uinmf, H6_uinmf), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_uinmf.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_uinmf))
    ari_uinmf.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_uinmf))
print('NMI (UINMF): {:.3f}'.format(max(nmi_uinmf)))
print('ARI (UINMF): {:.3f}'.format(max(ari_uinmf)))
resolution = np.arange(0.1, 10, 0.5)[np.argmax(nmi_uinmf)]
print(f"the best resolution: {resolution}")
labels_tmp = utils.leiden_cluster(X = np.concatenate((H1_uinmf, H2_uinmf, H3_uinmf, H4_uinmf, H5_uinmf, H6_uinmf), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)

uinmf_umap = UMAP(n_components = 2, min_dist = 0.4, random_state = 0).fit_transform(np.concatenate((H1_uinmf, H2_uinmf, H3_uinmf, H4_uinmf, H5_uinmf, H6_uinmf), axis = 0))
uinmf_umaps = []
leiden_labels_uinmf = []
for batch in range(n_batches):
    if batch == 0:
        start_pointer = 0
        end_pointer = start_pointer + zs[batch].shape[0]
        uinmf_umaps.append(uinmf_umap[start_pointer:end_pointer,:])
        leiden_labels_uinmf.append(labels_tmp[start_pointer:end_pointer])
    elif batch == (n_batches - 1):
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        uinmf_umaps.append(uinmf_umap[start_pointer:,:])
        leiden_labels_uinmf.append(labels_tmp[start_pointer:])
    else:
        start_pointer = start_pointer + zs[batch - 1].shape[0]
        end_pointer = start_pointer + zs[batch].shape[0]
        uinmf_umaps.append(uinmf_umap[start_pointer:end_pointer,:])
        leiden_labels_uinmf.append(labels_tmp[start_pointer:end_pointer])

utils.plot_latent_ext(uinmf_umaps, annos = labels, mode = "separate", save = uinmf_path + f'latent_separate_uinmf.png', 
                      figsize = (7,20), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", alpha = 0.7)

utils.plot_latent_ext(uinmf_umaps, annos = labels, mode = "modality", save = uinmf_path + f'latent_batches_uinmf.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, text_size = "large", alpha = 0.7)

utils.plot_latent_ext(uinmf_umaps, annos = labels, mode = "joint", save = uinmf_path + f'latent_joint_uinmf.png', 
                      figsize = (7,5), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = False, text_size = "large", alpha = 0.7)

# In[]
# calculate mapping of leiden cluster results and ground truth clusters
cluster_mapping_uinmf = {}
# only considering the last 3 batches, which includes scRNA-seq
uniq_clusters = np.unique(np.concatenate(leiden_labels_uinmf[3:]))
for cluster in uniq_clusters:
    cluster_idx = np.where(np.concatenate(leiden_labels_uinmf[3:]) == cluster)[0]
    gt_clusters, cell_counts = np.unique(np.concatenate(labels[3:])[cluster_idx], return_counts = True)
    cluster_mapping_uinmf[cluster] = gt_clusters[np.argsort(cell_counts)[-1]]

# TODO: Wilcoxon rank sum test for UINMF
wilcoxon_scores = pd.DataFrame(index = counts["feats_name"]["rna"])
for cluster in uniq_clusters:
    counts_cluster = np.concatenate(counts["rna"][3:], axis = 0)[np.concatenate(leiden_labels_uinmf[3:]) == cluster,:]
    counts_other = np.concatenate(counts["rna"][3:], axis = 0)[np.concatenate(leiden_labels_uinmf[3:]) != cluster,:]
    pvals = bmk.wilcoxon_rank_sum(counts_x = counts_cluster, counts_y = counts_other, fdr = False)
    wilcoxon_scores["cluster_" + str(cluster)] = pvals[:, None]

wilcoxon_scores.to_csv(scmomat_dir + "score_genes_wilcoxon.csv")

kt_scores_uinmf = []
for cluster in uniq_clusters:
    gt_cluster = cluster_mapping_uinmf[cluster]
    # rank based comparison
    logFC = de_analysis.loc[de_analysis["pop"] == gt_cluster, "logFC_theoretical"].values.squeeze()
    gene_enrich = np.where(logFC > 0)[0]

    scores_gt = de_analysis.loc[de_analysis["pop"] == gt_cluster, "wil.p_theoretical"].values.squeeze()
    # scores_gt = de_analysis.loc[de_analysis["pop"] == gt_cluster, "wil.p_true_counts"].values.squeeze()
    kt_score, _ = stats.kendalltau(wilcoxon_scores.loc[:, "cluster_" + str(cluster)].squeeze()[gene_enrich], scores_gt[gene_enrich])
    kt_scores_uinmf.append(kt_score)
    print(kt_score)


kt_scores = pd.DataFrame(columns = ["methods", "Kendall-tau"])
kt_scores["Kendall-tau"] = np.array(kt_scores_scmomat + kt_scores_uinmf)
kt_scores["methods"] = np.array(["scMoMaT"] * len(kt_scores_scmomat) + ["Wilcoxon"] * len(kt_scores_uinmf))
kt_scores.to_csv(scmomat_dir + "kt_scores.csv")

# In[]
for seed in [1,2,3,4,5]:
    result_dir = f"simulated/de_test_{seed}/scenario2"
    scmomat_dir = result_dir + "/scmomat/"
    kt_score = pd.read_csv(scmomat_dir + "kt_scores.csv", index_col = 0)
    if seed == 1:
        kt_scores = kt_score
    else:
        kt_scores = pd.concat([kt_scores, kt_score])

fig = plt.figure(figsize = (4,5))
ax = fig.add_subplot()
import seaborn as sns
plt.rcParams["font.size"] = 20
sns.barplot(data = kt_scores, x = "methods", y = "Kendall-tau", ax = ax, capsize = 0.3)
# more than 10 points
# sns.stripplot(data = kt_scores, x = "methods", y = "Kendall-tau", ax = ax, color = "black")    
# for i in ax.containers:
#     ax.bar_label(i, color = "b")
# fig.savefig("simulated/de_test.png", bbox_inches = "tight")

# %%
