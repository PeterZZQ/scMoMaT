# In[]
import sys, os
sys.path.append('../')
sys.path.append('../src/')
import torch
import numpy as np
import umap_batch
from umap import UMAP
import pandas as pd  
import scipy.sparse as sp
from scipy.io import mmwrite

import model
import time
import bmk
import utils

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# In[] 
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   NOTE: 1. Load dataset, sub-sampling and running scmomat (without retraining, retraining see the third section)
#
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# NOTE: subsample B cells, and plot pie chart 
# read in dataset
dir = '../data/real/diag/Xichen/'
remove_dir = '../data/real/diag/Xichen/remove_celltype/'

result_dir_removecelltype = "spleen/remove_celltype/scmomat/"

# original data matrix
counts_rnas = []
counts_atacs = []
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
    counts_rnas.append(counts_rna)
    counts_atacs.append(counts_atac)

counts = {"rna":counts_rnas, "atac": counts_atacs}

# NOTE: remove cell types, the post-processing will force the unique cell type in one batch to find neighbors in another batch, which doesn't correspond to the cell type.
# counts["atac"][1] = counts["atac"][1][(labels[1] != 'B_follicular') & (labels[1] != 'B_follicular_transitional') & (labels[1] != 'Marginal_Zone_B'), :]
# labels[1] = labels[1][(labels[1] != 'B_follicular') & (labels[1] != 'B_follicular_transitional') & (labels[1] != 'Marginal_Zone_B')]
# counts["rna"][0] = counts["rna"][0][(labels[0] != 'B_follicular') & (labels[0] != 'B_follicular_transitional') & (labels[0] != 'Marginal_Zone_B'), :] #  & (labels[0] != 'Unknown') & (labels[0] != 'Proliferating')
# labels[0] = labels[0][(labels[0] != 'B_follicular') & (labels[0] != 'B_follicular_transitional') & (labels[0] != 'Marginal_Zone_B')]
# make T_CD8_naive AND Memory_CD8_T to be all T_CD8
labels[0] = np.where(labels[0] == "T_CD8_naive", "T_CD8", labels[0])
labels[0] = np.where(labels[0] == "Memory_CD8_T", "T_CD8", labels[0])
labels[1] = np.where(labels[1] == "T_CD8_naive", "T_CD8", labels[1])
labels[1] = np.where(labels[1] == "Memory_CD8_T", "T_CD8", labels[1])

plt.rcParams["font.size"] = 15

fig = plt.figure(figsize = (10, 10))
axs = fig.subplots(ncols = 1, nrows = 2)
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
cell_types, sizes = np.unique(labels[0], return_counts = True)
colormap = plt.cm.get_cmap("Paired", len(cell_types))
percents = 100.*sizes/sizes.sum()
legends = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(cell_types, percents)]
legends = ['{0}'.format(i) for i in cell_types]

patches, texts = axs[0].pie(sizes, colors = colormap.colors, startangle=90)
axs[0].axis('equal')  

sort_legend = False
if sort_legend:
    patches, legends, dummy =  zip(*sorted(zip(patches, legends, sizes), key=lambda x: x[2], reverse=True))

axs[0].legend(patches, legends, loc='upper left', prop={'size': 15}, bbox_to_anchor=(1.04, 1.), fontsize=8)


sizes2 = []
for cell_type in cell_types:
    sizes2.append(np.where(labels[1] == cell_type)[0].shape[0])
sizes2 = np.array(sizes2)

percents = 100.*sizes2/sizes2.sum()
legends = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(cell_types, percents)]
legends = ['{0}'.format(i) for i in cell_types]

patches, texts = axs[1].pie(sizes2, colors = colormap.colors, startangle=90)
axs[0].axis('equal')  
axs[0].set_title("Batch 1")

sort_legend = False
if sort_legend:
    patches, legends, dummy =  zip(*sorted(zip(patches, legends, sizes2), key=lambda x: x[2], reverse=True))

axs[1].legend(patches, legends, loc='upper left', prop={'size': 15}, bbox_to_anchor=(1.04, 1.), fontsize=8)

axs[1].axis('equal')  
axs[1].set_title("Batch 2")
plt.tight_layout()
fig.savefig(result_dir_removecelltype + "pie_ori.pdf", bbox_inches = "tight")

# In[]
# NOTE: subsample B cells, and plot pie chart 
# only subsample cells in some cell types instead of totally remove them
np.random.seed(0)
counts_rnas = []
counts_atacs = []
labels = []
n_batches = 2
for batch in range(1, n_batches+1):
    meta_cell = pd.read_csv(os.path.join(dir, 'meta_c' + str(batch) + '.csv'), index_col=0)
    if batch == 1:
        # down sampling B_follicular, B_follicular_transitional, and Marginal_Zone_B
        barcodes1 = meta_cell.loc[(meta_cell["cell_type"] == "B_follicular") | (meta_cell["cell_type"] == "B_follicular_transitional") | (meta_cell["cell_type"] == "Marginal_Zone_B"), :].index.values.squeeze()
        barcodes1 = np.random.choice(barcodes1, size = 100, replace = False)
        barcodes2 = meta_cell.loc[(meta_cell["cell_type"] != "B_follicular") & (meta_cell["cell_type"] != "B_follicular_transitional") & (meta_cell["cell_type"] != "Marginal_Zone_B"), :].index.values.squeeze() # & (meta_cell["cell_type"] != 'Unknown') & (meta_cell["cell_type"] != 'Proliferating')
        meta_cell_sub = meta_cell.loc[np.concatenate((barcodes1, barcodes2), axis = 0), :]
        
        # barcodes1 = meta_cell.loc[(meta_cell["cell_type"] == "T_CD4_naive") | (meta_cell["cell_type"] == "T_CD4_reg") | (meta_cell["cell_type"] == "T_CD8_naive") | (meta_cell["cell_type"] == "Memory_CD8_T") , :].index.values.squeeze()
        # barcodes1 = np.random.choice(barcodes1, size = 1, replace = False)
        # barcodes2 = meta_cell.loc[(meta_cell["cell_type"] != "T_CD4_naive") & (meta_cell["cell_type"] != "T_CD4_reg") & (meta_cell["cell_type"] != "T_CD8_naive") & (meta_cell["cell_type"] != "Memory_CD8_T") , :].index.values.squeeze() 
        # meta_cell_sub = meta_cell.loc[np.concatenate((barcodes1, barcodes2), axis = 0), :]
    else:
        # # down sampling B_follicular, B_follicular_transitional, and Marginal_Zone_B
        # barcodes1 = meta_cell.loc[(meta_cell["cell_type"] == "T_CD8_naive") | (meta_cell["cell_type"] == "Memory_CD8_T") | (meta_cell["cell_type"] == "T_CD4_naive") | (meta_cell["cell_type"] == "T_CD4_reg"), :].index.values.squeeze()
        # print(barcodes1.shape)
        # barcodes1 = np.random.choice(barcodes1, size = 200, replace = False)
        # print(barcodes1.shape)
        # barcodes2 = meta_cell.loc[(meta_cell["cell_type"] != "T_CD8_naive") & (meta_cell["cell_type"] != "Memory_CD8_T") & (meta_cell["cell_type"] != "T_CD4_naive") & (meta_cell["cell_type"] != 'T_CD4_reg'), :].index.values.squeeze()
        # meta_cell_sub = meta_cell.loc[np.concatenate((barcodes1, barcodes2), axis = 0), :]
    
        meta_cell_sub = meta_cell

    print(meta_cell_sub.shape)
    meta_cell_sub.to_csv(remove_dir + "meta_c" + str(batch) + ".csv")
    labels.append(meta_cell_sub["cell_type"].values.squeeze())

    try:
        counts_atac = np.array(sp.load_npz(os.path.join(dir, 'RxC' + str(batch) + ".npz")).todense().T)
        if batch == 2:
            counts_atac = pd.DataFrame(data = counts_atac, index = meta_cell.index.values.squeeze())
            counts_atac = counts_atac.loc[meta_cell_sub.index.values, :].values
            sp.save_npz(remove_dir + "RxC" + str(batch) + ".npz", sp.csr_matrix(counts_atac.T))   
            mmwrite(remove_dir + "RxC" + str(batch) + ".mtx", sp.csr_matrix(counts_atac.T))
        else:    
            sp.save_npz(remove_dir + "RxC" + str(batch) + ".npz", sp.csr_matrix(counts_atac.T))
            mmwrite(remove_dir + "RxC" + str(batch) + ".mtx", sp.csr_matrix(counts_atac.T))

        counts_atac = utils.preprocess(counts_atac, modality = "ATAC")
    except:
        counts_atac = None        
    try:
        counts_rna = np.array(sp.load_npz(os.path.join(dir, 'GxC' + str(batch) + ".npz")).todense().T)
        if batch == 1:
            counts_rna = pd.DataFrame(data = counts_rna, index = meta_cell.index.values.squeeze())
            counts_rna = counts_rna.loc[meta_cell_sub.index.values, :].values
            print(batch)
            print(counts_rna.shape)
            sp.save_npz(remove_dir + "GxC" + str(batch) + ".npz", sp.csr_matrix(counts_rna.T))
            mmwrite(remove_dir + "GxC" + str(batch) + ".mtx", sp.csr_matrix(counts_rna.T))
        else:
            print(batch)
            print(counts_rna.shape)
            sp.save_npz(remove_dir + "GxC" + str(batch) + ".npz", sp.csr_matrix(counts_rna.T))
            mmwrite(remove_dir + "GxC" + str(batch) + ".mtx", sp.csr_matrix(counts_rna.T))

        counts_rna = utils.preprocess(counts_rna, modality = "RNA", log = False)
    except:
        counts_rna = None
    counts_rnas.append(counts_rna)
    counts_atacs.append(counts_atac)

counts = {"rna":counts_rnas, "atac": counts_atacs}


A = sp.load_npz(os.path.join(dir, 'GxR.npz'))
A = np.array(A.todense())
interacts = None

genes = pd.read_csv(dir + "genes.txt", header = None).values.squeeze()
regions = pd.read_csv(dir + "regions.txt", header = None).values.squeeze()

feats_name = {"rna": genes, "atac": regions}
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

# make T_CD8_naive AND Memory_CD8_T to be all T_CD8
labels[0] = np.where(labels[0] == "T_CD8_naive", "T_CD8", labels[0])
labels[0] = np.where(labels[0] == "Memory_CD8_T", "T_CD8", labels[0])
labels[1] = np.where(labels[1] == "T_CD8_naive", "T_CD8", labels[1])
labels[1] = np.where(labels[1] == "Memory_CD8_T", "T_CD8", labels[1])
# rna batch (batch 1) unique Unknown, and Proliferating


fig = plt.figure(figsize = (10, 10))
axs = fig.subplots(ncols = 1, nrows = 2)
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
cell_types, sizes = np.unique(labels[0], return_counts = True)
colormap = plt.cm.get_cmap("Paired", len(cell_types))
percents = 100.*sizes/sizes.sum()
legends = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(cell_types, percents)]
legends = ['{0}'.format(i) for i in cell_types]

patches, texts = axs[0].pie(sizes, colors = colormap.colors, startangle=90)
axs[0].axis('equal')  

sort_legend = False
if sort_legend:
    patches, legends, dummy =  zip(*sorted(zip(patches, legends, sizes), key=lambda x: x[2], reverse=True))

axs[0].legend(patches, legends, loc='upper left', prop={'size': 15}, bbox_to_anchor=(1.04, 1.), fontsize=8)

sizes2 = []
for cell_type in cell_types:
    sizes2.append(np.where(labels[1] == cell_type)[0].shape[0])
sizes2 = np.array(sizes2)
percents = 100.*sizes2/sizes2.sum()
legends = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(cell_types, percents)]
legends = ['{0}'.format(i) for i in cell_types]

patches, texts = axs[1].pie(sizes2, colors = colormap.colors, startangle=90)
axs[0].axis('equal')
axs[0].set_title("Batch 1")  

sort_legend = False
if sort_legend:
    patches, legends, dummy =  zip(*sorted(zip(patches, legends, sizes2), key=lambda x: x[2], reverse=True))

axs[1].legend(patches, legends, loc='upper left', prop={'size': 15}, bbox_to_anchor=(1.04, 1.), fontsize=8)

axs[1].axis('equal')  
axs[1].set_title("Batch 2")
plt.tight_layout()
fig.savefig(result_dir_removecelltype + "pie_sub.pdf", bbox_inches = "tight")

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

x = np.linspace(0, T, int(T/interval) + 1)
plt.plot(x, losses1)

# torch.save(model1, result_dir_removecelltype + f'CFRM_{K}_{T}.pt')
# model1 = torch.load(result_dir_removecelltype + f'CFRM_{K}_{T}.pt')

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

utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = result_dir_removecelltype + f'latent_separate_{K}_{T}.png', figsize = (10,15), axis_label = "UMAP")

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = result_dir_removecelltype + f'latent_batches_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6)

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = result_dir_removecelltype + f'latent_clusters_{K}_{T}.png', figsize = (15,10), axis_label = "UMAP", markerscale = 6)


# In[] 
# NOTE: Post-processing, clustering, and plot the result after post-processing
plt.rcParams["font.size"] = 10
n_neighbors = 30
r = None

zs = []
for batch in range(n_batches):
    z = model1.softmax(model1.C_cells[str(batch)].cpu().detach()).numpy()
    zs.append(z)
s_pair_dist, knn_indices, knn_dists = utils.post_process(zs, n_neighbors, njobs = 8, r = r)
# here load the score.csv that we calculated in advance to select the best resolution
resolution = 0.6

labels_tmp = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
umap_op = umap_batch.UMAP(n_components = 2, n_neighbors = n_neighbors, min_dist = 0.2, random_state = 0, 
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

utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = result_dir_removecelltype + f'latent_separate_{K}_{T}_processed2.png', 
                      figsize = (10,12), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

utils.plot_latent_ext(x_umaps, annos = labels, mode = "modality", save = result_dir_removecelltype + f'latent_batches_{K}_{T}_processed2.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

utils.plot_latent_ext(x_umaps, annos = labels, mode = "joint", save = result_dir_removecelltype + f'latent_clusters_{K}_{T}_processed2.png', 
                      figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

utils.plot_latent_ext(x_umaps, annos = leiden_labels, mode = "joint", save = result_dir_removecelltype + f'latent_leiden_clusters_{K}_{T}_{resolution}_processed2.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)


# In[]
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   NOTE: 2. Benchmarking with baseline methods
#
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# NOTE: Baseline methods
# 1. UINMF
uinmf_path = "spleen/remove_celltype/uinmf_bin/" 
H1_uinmf = pd.read_csv(uinmf_path + "liger_c1_norm.csv", index_col = 0).values
H2_uinmf = pd.read_csv(uinmf_path + "liger_c2_norm.csv", index_col = 0).values
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
                      figsize = (15,15), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

utils.plot_latent_ext(uinmf_umaps, annos = labels, mode = "modality", save = uinmf_path + f'latent_batches_uinmf.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

utils.plot_latent_ext(uinmf_umaps, annos = labels, mode = "joint", save = uinmf_path + f'latent_clusters_uinmf.png', 
                      figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)



# 2. Multimap
multimap_path = "spleen/remove_celltype/multimap/"
batches = pd.read_csv(multimap_path + "batch_id.csv", index_col = 0)
X_multimap = np.load(multimap_path + "multimap.npy")
G_multimap = sp.load_npz(multimap_path + "multimap_graph.npz").todense()
X_multimaps = []
for batch in ["RNA", "ATAC"]:
    X_multimaps.append(X_multimap[batches.values.squeeze() == batch, :])

utils.plot_latent_ext(X_multimaps, annos = labels, mode = "separate", save = multimap_path + f'latent_separate_multimap.png', 
                      figsize = (15,15), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

utils.plot_latent_ext(X_multimaps, annos = labels, mode = "modality", save = multimap_path + f'latent_batches_multimap.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

utils.plot_latent_ext(X_multimaps, annos = labels, mode = "joint", save = multimap_path + f'latent_clusters_multimap.png', 
                      figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

# 3. Seurat
seurat_path = "spleen/remove_celltype/seurat/"
seurat_pcas = [pd.read_csv(seurat_path + "seurat_pca_c1.txt", sep = "\t", index_col = 0).values, 
               pd.read_csv(seurat_path + "seurat_pca_c2.txt", sep = "\t", index_col = 0).values]
seurat_umaps = [pd.read_csv(seurat_path + "seurat_umap_c1.txt", sep = "\t", index_col = 0).values,
               pd.read_csv(seurat_path + "seurat_umap_c2.txt", sep = "\t", index_col = 0).values]


utils.plot_latent_ext(seurat_umaps, annos = labels, mode = "separate", save = seurat_path + f'latent_separate_seurat.png', 
                      figsize = (15,15), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

utils.plot_latent_ext(seurat_umaps, annos = labels, mode = "modality", save = seurat_path + f'latent_batches_seurat.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

utils.plot_latent_ext(seurat_umaps, annos = labels, mode = "joint", save = seurat_path + f'latent_clusters_seurat.png', 
                      figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)


# 4. Liger
liger_path = "spleen/remove_celltype/liger/"
H1_liger = pd.read_csv(liger_path + "liger_c1_norm.csv", sep = ",", index_col = 0).values
H2_liger = pd.read_csv(liger_path + "liger_c2_norm.csv", sep = ",", index_col = 0).values
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
                      figsize = (15,15), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

utils.plot_latent_ext(liger_umaps, annos = labels, mode = "modality", save = liger_path + f'latent_batches_liger.png', 
                      figsize = (10,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

utils.plot_latent_ext(liger_umaps, annos = labels, mode = "joint", save = liger_path + f'latent_clusters_liger.png', 
                      figsize = (12,7), axis_label = "UMAP", markerscale = 6, s = 5, label_inplace = True, text_size = "large", colormap = "Paired", alpha = 0.7)

# In[]
# graph connectivity score (gc) measure the batch effect removal per cell identity
# 1. scMoMaT
# construct neighborhood graph from the post-processed latent space
knn_graph = np.zeros((knn_indices.shape[0], knn_indices.shape[0]))
knn_graph[np.arange(knn_indices.shape[0])[:, None], knn_indices] = 1
gc_scmomat = bmk.graph_connectivity(G = knn_graph, groups = np.concatenate(labels, axis = 0))
print('GC (scmomat): {:.3f}'.format(gc_scmomat))

# 2. Seurat, n_neighbors affect the overall acc, and should be the same as scJMT
n_neighbors = knn_indices.shape[1]
gc_seurat = bmk.graph_connectivity(X = np.concatenate(seurat_pcas, axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (Seurat): {:.3f}'.format(gc_seurat))

# 3. Liger
gc_liger = bmk.graph_connectivity(X = np.concatenate((H1_liger, H2_liger), axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (Liger): {:.3f}'.format(gc_liger))

# 4. UINMF
gc_uinmf = bmk.graph_connectivity(X = np.concatenate((H1_uinmf, H2_uinmf), axis = 0), groups = np.concatenate(labels, axis = 0), k = n_neighbors)
print('GC (UINMF): {:.3f}'.format(gc_uinmf))

# 5. Multimap
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
    leiden_labels_scjmt = utils.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)
    nmi_scmomat.append(bmk.nmi(group1 = np.concatenate(labels), group2 = leiden_labels_scjmt))
    ari_scmomat.append(bmk.ari(group1 = np.concatenate(labels), group2 = leiden_labels_scjmt))
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

scores = pd.DataFrame(columns = ["methods", "resolution", "NMI", "ARI", "GC"])
scores["NMI"] = np.array(nmi_scmomat + nmi_seurat + nmi_liger + nmi_uinmf + nmi_multimap)
scores["ARI"] = np.array(ari_scmomat + ari_seurat + ari_liger + ari_uinmf + ari_multimap)
scores["GC"] = np.array([gc_scmomat] * len(nmi_scmomat) + [gc_seurat] * len(nmi_seurat) + [gc_liger] * len(nmi_liger) + [gc_uinmf] * len(nmi_uinmf) + [gc_multimap] * len(nmi_multimap))
scores["resolution"] = np.array([x for x in np.arange(0.1, 10, 0.5)] * 5)
scores["methods"] = np.array(["scMoMaT"] * len(nmi_scmomat) + ["Seurat"] * len(nmi_seurat) + ["Liger"] * len(nmi_liger) + ["UINMF"] * len(nmi_uinmf) + ["MultiMap"] * len(nmi_multimap))
scores.to_csv(result_dir_removecelltype + "scores.csv")

# In[]
# score for post_nn_distance2
scores = pd.read_csv(result_dir_removecelltype + "scores.csv", index_col = 0)

print("GC (scMoMaT) postprocess 2: {:.4f}".format(np.max(scores.loc[scores["methods"] == "scMoMaT", "GC"].values)))
print("NMI (scMoMaT) postprocess 2: {:.4f}".format(np.max(scores.loc[scores["methods"] == "scMoMaT", "NMI"].values)))
print("ARI (scMoMaT) postprocess 2: {:.4f}".format(np.max(scores.loc[scores["methods"] == "scMoMaT", "ARI"].values)))

print("GC (UINMF): {:.4f}".format(np.max(scores.loc[scores["methods"] == "UINMF", "GC"].values)))
print("NMI (UINMF): {:.4f}".format(np.max(scores.loc[scores["methods"] == "UINMF", "NMI"].values)))
print("ARI (UINMF): {:.4f}".format(np.max(scores.loc[scores["methods"] == "UINMF", "ARI"].values)))

print("GC (MultiMap): {:.4f}".format(np.max(scores.loc[scores["methods"] == "MultiMap", "GC"].values)))
print("NMI (MultiMap): {:.4f}".format(np.max(scores.loc[scores["methods"] == "MultiMap", "NMI"].values)))
print("ARI (MultiMap): {:.4f}".format(np.max(scores.loc[scores["methods"] == "MultiMap", "ARI"].values)))

print("GC (LIGER): {:.4f}".format(np.max(scores.loc[scores["methods"] == "Liger", "GC"].values)))
print("NMI (LIGER): {:.4f}".format(np.max(scores.loc[scores["methods"] == "Liger", "NMI"].values)))
print("ARI (LIGER): {:.4f}".format(np.max(scores.loc[scores["methods"] == "Liger", "ARI"].values)))

print("GC (Seurat): {:.4f}".format(np.max(scores.loc[scores["methods"] == "Seurat", "GC"].values)))
print("NMI (Seurat): {:.4f}".format(np.max(scores.loc[scores["methods"] == "Seurat", "NMI"].values)))
print("ARI (Seurat): {:.4f}".format(np.max(scores.loc[scores["methods"] == "Seurat", "ARI"].values)))


scores2 = pd.read_csv("spleen/scmomat/scores.csv", index_col = 0)

print("GC (scMoMaT) postprocess 2: {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "scMoMaT", "GC"].values)))
print("NMI (scMoMaT) postprocess 2: {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "scMoMaT", "NMI"].values)))
print("ARI (scMoMaT) postprocess 2: {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "scMoMaT", "ARI"].values)))

print("GC (UINMF): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "UINMF", "GC"].values)))
print("NMI (UINMF): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "UINMF", "NMI"].values)))
print("ARI (UINMF): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "UINMF", "ARI"].values)))

print("GC (MultiMap): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "MultiMap", "GC"].values)))
print("NMI (MultiMap): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "MultiMap", "NMI"].values)))
print("ARI (MultiMap): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "MultiMap", "ARI"].values)))

print("GC (LIGER): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Liger", "GC"].values)))
print("NMI (LIGER): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Liger", "NMI"].values)))
print("ARI (LIGER): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Liger", "ARI"].values)))

print("GC (Seurat): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Seurat", "GC"].values)))
print("NMI (Seurat): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Seurat", "NMI"].values)))
print("ARI (Seurat): {:.4f}".format(np.max(scores2.loc[scores2["methods"] == "Seurat", "ARI"].values)))



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

gc_scmomat = np.max(scores.loc[scores["methods"] == "scMoMaT", "GC"].values)
gc_uinmf = np.max(scores.loc[scores["methods"] == "UINMF", "GC"].values)
gc_multimap = np.max(scores.loc[scores["methods"] == "MultiMap", "GC"].values)
gc_liger = np.max(scores.loc[scores["methods"] == "Liger", "GC"].values)

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3,4], [gc_scmomat, gc_uinmf, gc_multimap, gc_liger], width = 0.4)
barlist[0].set_color('r')
fig.savefig(result_dir_removecelltype + "GC.pdf", bbox_inches = "tight")    

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_title("graph connectivity", fontsize = 20)
_ = ax.set_xticks([1,2,3,4])
_ = ax.set_xticklabels(["scMoMaT", "UINMF", "MultiMap", "Liger"])
# _ = ax.set_xlabel("cluster", fontsize = 20)
_ = ax.set_ylabel("GC", fontsize = 20)
show_values_on_bars(ax)
fig.savefig(result_dir_removecelltype + "GC.png", bbox_inches = "tight")    

# NMI
nmi_scmomat = np.max(scores.loc[scores["methods"] == "scMoMaT", "NMI"].values)
nmi_uinmf = np.max(scores.loc[scores["methods"] == "UINMF", "NMI"].values)
nmi_multimap = np.max(scores.loc[scores["methods"] == "MultiMap", "NMI"].values)
nmi_liger = np.max(scores.loc[scores["methods"] == "Liger", "NMI"].values)

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3,4], [nmi_scmomat, nmi_uinmf, nmi_multimap, nmi_liger], width = 0.4)
barlist[0].set_color('r')    

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_title("NMI", fontsize = 20)
_ = ax.set_xticks([1,2,3,4])
_ = ax.set_xticklabels(["scMoMaT", "UINMF", "MultiMap", "Liger"])
# _ = ax.set_xlabel("cluster", fontsize = 20)
_ = ax.set_ylabel("NMI", fontsize = 20)
show_values_on_bars(ax)
fig.savefig(result_dir_removecelltype + "NMI.png", bbox_inches = "tight")    

# ARI
ari_scmomat = np.max(scores.loc[scores["methods"] == "scMoMaT", "ARI"].values)
ari_uinmf = np.max(scores.loc[scores["methods"] == "UINMF", "ARI"].values)
ari_multimap = np.max(scores.loc[scores["methods"] == "MultiMap", "ARI"].values)
ari_liger = np.max(scores.loc[scores["methods"] == "Liger", "ARI"].values)

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
fig.savefig(result_dir_removecelltype + "ARI.png", bbox_inches = "tight")    

# %%
