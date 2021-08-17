# In[0]
import sys, os
sys.path.append('../')
sys.path.append('../src/')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from umap import UMAP
import pandas as pd 
import scipy.sparse as sp
import torch
import time

import model
import utils
import quantile 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# To be added, include the sparse model, check the biclustering result on scRNA-Seq, scATAC-Seq and protein

# In[1] read data
dir = '../data/real/ASAP-PBMC/'

counts_rnas = []
counts_atacs = []
counts_proteins = []
for batch in range(1,5):
    try:
        counts_atac = np.array(sp.load_npz(os.path.join(dir, 'RxC' + str(batch) + ".npz")).todense().T)
        counts_atac = utils.preprocess(counts_atac, modality = "ATAC")
    except:
        counts_atac = None
        
    try:
        counts_rna = np.array(sp.load_npz(os.path.join(dir, 'GxC' + str(batch) + ".npz")).todense().T)
        counts_rna = utils.preprocess(counts_rna, modality = "RNA")
    except:
        counts_rna = None
    
    try:
        counts_protein = np.array(sp.load_npz(os.path.join(dir, 'PxC' + str(batch) + ".npz")).todense().T)
        counts_protein = utils.preprocess(counts_protein, modality = "RNA")
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

interacts = {"rna_atac": A2, "rna_protein": A1}

# hyper parameters
Ns = [10] * 4
K = 10
N_feat = Ns[0] + 1
interval = 20000
T = 10000
lr = 1e-3
run = 0
batchsize = 0.3

print("not using interaction")
print("entropy")

for alpha in ([[1000, 1, 100, 100, 0.00], [1000, 1, 100, 100, 0.10], [1000, 1, 100, 100, 0.50], [1000, 1, 100, 100, 1.00]]):
    print("alpha: " + str(alpha))
    model1 = model.cfrm_new(counts = counts, interacts = None, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
    losses1 = model1.train_func(T = T)
    model1.assign_clusters(relocate_empty = True, n_relocate = None)

    # calculate the assignment accuracy 
    within_connects1 = np.zeros_like(interacts["rna_atac"])
    within_connects2 = np.zeros_like(interacts["rna_protein"])
        
    for clust in range(model1.binary_C_feats[0].shape[1]):
        clust_feats_rna = np.where(model1.binary_C_feats[0][:,clust] == True)[0]
        clust_feats_atac = np.where(model1.binary_C_feats[1][:,clust] == True)[0]
        clust_feats_protein = np.where(model1.binary_C_feats[2][:,clust] == True)[0]
        
        within_connects1[np.ix_(clust_feats_rna, clust_feats_atac)] = 1
        within_connects2[np.ix_(clust_feats_rna, clust_feats_protein)] = 1
        
    correct_conn = np.sum(within_connects1 * interacts["rna_atac"])/np.sum(within_connects1)
    false_conn = np.sum(((1 - within_connects1)) * interacts["rna_atac"])/np.sum(1 - within_connects1)
    print("ATAC_RNA")
    # Precision, true positive/(true positive + false positive)
    print("Precision: " + str(correct_conn))
    # False omission rate (FOR): false negative/(false negative + true negative)
    print("FOR: " + str(false_conn))
    print("Ratio: " + str(correct_conn/false_conn))
    print()

    correct_conn = np.sum(within_connects2 * interacts["rna_protein"])/np.sum(within_connects1)
    false_conn = np.sum(((1 - within_connects2)) * interacts["rna_protein"])/np.sum(1 - within_connects1)
    print("PROTEIN_RNA")
    print("Precision: " + str(correct_conn))
    print("FOR: " + str(false_conn))
    print("Ratio: " + str(correct_conn/false_conn))
    print()

# hyper parameters
Ns = [10] * 4
K = 10
N_feat = Ns[0] + 1
interval = 20000
T = 10000
lr = 1e-3
run = 0
batchsize = 0.3

print("orthogonality")
for alpha in ([[1000, 1, 100, 100, 0.00], [1000, 1, 100, 100, 0.02], [1000, 1, 100, 100, 0.50], [1000, 1, 100, 100, 0.10]]):
    print("alpha: " + str(alpha))
    model1 = model.cfrm_new2(counts = counts, interacts = None, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
    losses1 = model1.train_func(T = T)
    model1.assign_clusters(relocate_empty = True, n_relocate = None)

    # calculate the assignment accuracy 
    within_connects1 = np.zeros_like(interacts["rna_atac"])
    within_connects2 = np.zeros_like(interacts["rna_protein"])
        
    for clust in range(model1.binary_C_feats[0].shape[1]):
        clust_feats_rna = np.where(model1.binary_C_feats[0][:,clust] == True)[0]
        clust_feats_atac = np.where(model1.binary_C_feats[1][:,clust] == True)[0]
        clust_feats_protein = np.where(model1.binary_C_feats[2][:,clust] == True)[0]
        
        within_connects1[np.ix_(clust_feats_rna, clust_feats_atac)] = 1
        within_connects2[np.ix_(clust_feats_rna, clust_feats_protein)] = 1
        
    correct_conn = np.sum(within_connects1 * interacts["rna_atac"])/np.sum(within_connects1)
    false_conn = np.sum(((1 - within_connects1)) * interacts["rna_atac"])/np.sum(1 - within_connects1)
    print("ATAC_RNA")
    # Precision, true positive/(true positive + false positive)
    print("Precision: " + str(correct_conn))
    # False omission rate (FOR): false negative/(false negative + true negative)
    print("FOR: " + str(false_conn))
    print("Ratio: " + str(correct_conn/false_conn))
    print()

    correct_conn = np.sum(within_connects2 * interacts["rna_protein"])/np.sum(within_connects1)
    false_conn = np.sum(((1 - within_connects2)) * interacts["rna_protein"])/np.sum(1 - within_connects1)
    print("PROTEIN_RNA")
    print("Precision: " + str(correct_conn))
    print("FOR: " + str(false_conn))
    print("Ratio: " + str(correct_conn/false_conn))
    print()

print("sparsemax")
for alpha in ([[1000, 1, 100, 100, 0.00], [1000, 1, 100, 100, 0.02], [1000, 1, 100, 100, 0.50], [1000, 1, 100, 100, 0.10]]):
    print("alpha: " + str(alpha))
    model1 = model.cfrm_new3(counts = counts, interacts = None, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
    losses1 = model1.train_func(T = T)
    model1.assign_clusters(relocate_empty = True, n_relocate = None)

    # calculate the assignment accuracy 
    within_connects1 = np.zeros_like(interacts["rna_atac"])
    within_connects2 = np.zeros_like(interacts["rna_protein"])
        
    for clust in range(model1.binary_C_feats[0].shape[1]):
        clust_feats_rna = np.where(model1.binary_C_feats[0][:,clust] == True)[0]
        clust_feats_atac = np.where(model1.binary_C_feats[1][:,clust] == True)[0]
        clust_feats_protein = np.where(model1.binary_C_feats[2][:,clust] == True)[0]
        
        within_connects1[np.ix_(clust_feats_rna, clust_feats_atac)] = 1
        within_connects2[np.ix_(clust_feats_rna, clust_feats_protein)] = 1
        
    correct_conn = np.sum(within_connects1 * interacts["rna_atac"])/np.sum(within_connects1)
    false_conn = np.sum(((1 - within_connects1)) * interacts["rna_atac"])/np.sum(1 - within_connects1)
    print("ATAC_RNA")
    # Precision, true positive/(true positive + false positive)
    print("Precision: " + str(correct_conn))
    # False omission rate (FOR): false negative/(false negative + true negative)
    print("FOR: " + str(false_conn))
    print("Ratio: " + str(correct_conn/false_conn))
    print()

    correct_conn = np.sum(within_connects2 * interacts["rna_protein"])/np.sum(within_connects1)
    false_conn = np.sum(((1 - within_connects2)) * interacts["rna_protein"])/np.sum(1 - within_connects1)
    print("PROTEIN_RNA")
    print("Precision: " + str(correct_conn))
    print("FOR: " + str(false_conn))
    print("Ratio: " + str(correct_conn/false_conn))
    print()

print("using interaction")
print("entropy")
for alpha in ([[1000, 1, 100, 100, 0.00], [1000, 1, 100, 100, 0.10], [1000, 1, 100, 100, 0.50], [1000, 1, 100, 100, 1.00]]):
    print("alpha: " + str(alpha))
    model1 = model.cfrm_new(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
    losses1 = model1.train_func(T = T)
    model1.assign_clusters(relocate_empty = True, n_relocate = None)

    # calculate the assignment accuracy 
    within_connects1 = np.zeros_like(interacts["rna_atac"])
    within_connects2 = np.zeros_like(interacts["rna_protein"])
        
    for clust in range(model1.binary_C_feats[0].shape[1]):
        clust_feats_rna = np.where(model1.binary_C_feats[0][:,clust] == True)[0]
        clust_feats_atac = np.where(model1.binary_C_feats[1][:,clust] == True)[0]
        clust_feats_protein = np.where(model1.binary_C_feats[2][:,clust] == True)[0]
        
        within_connects1[np.ix_(clust_feats_rna, clust_feats_atac)] = 1
        within_connects2[np.ix_(clust_feats_rna, clust_feats_protein)] = 1
        
    correct_conn = np.sum(within_connects1 * interacts["rna_atac"])/np.sum(within_connects1)
    false_conn = np.sum(((1 - within_connects1)) * interacts["rna_atac"])/np.sum(1 - within_connects1)
    print("ATAC_RNA")
    # Precision, true positive/(true positive + false positive)
    print("Precision: " + str(correct_conn))
    # False omission rate (FOR): false negative/(false negative + true negative)
    print("FOR: " + str(false_conn))
    print("Ratio: " + str(correct_conn/false_conn))
    print()

    correct_conn = np.sum(within_connects2 * interacts["rna_protein"])/np.sum(within_connects1)
    false_conn = np.sum(((1 - within_connects2)) * interacts["rna_protein"])/np.sum(1 - within_connects1)
    print("PROTEIN_RNA")
    print("Precision: " + str(correct_conn))
    print("FOR: " + str(false_conn))
    print("Ratio: " + str(correct_conn/false_conn))
    print()


# hyper parameters
Ns = [10] * 4
K = 10
N_feat = Ns[0] + 1
interval = 20000
T = 10000
lr = 1e-3
run = 0
batchsize = 0.3

print("orthogonality")
for alpha in ([[1000, 1, 100, 100, 0.00], [1000, 1, 100, 100, 0.02], [1000, 1, 100, 100, 0.50], [1000, 1, 100, 100, 0.10]]):
    print("alpha: " + str(alpha))
    model1 = model.cfrm_new2(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
    losses1 = model1.train_func(T = T)
    model1.assign_clusters(relocate_empty = True, n_relocate = None)

    # calculate the assignment accuracy 
    within_connects1 = np.zeros_like(interacts["rna_atac"])
    within_connects2 = np.zeros_like(interacts["rna_protein"])
        
    for clust in range(model1.binary_C_feats[0].shape[1]):
        clust_feats_rna = np.where(model1.binary_C_feats[0][:,clust] == True)[0]
        clust_feats_atac = np.where(model1.binary_C_feats[1][:,clust] == True)[0]
        clust_feats_protein = np.where(model1.binary_C_feats[2][:,clust] == True)[0]
        
        within_connects1[np.ix_(clust_feats_rna, clust_feats_atac)] = 1
        within_connects2[np.ix_(clust_feats_rna, clust_feats_protein)] = 1
        
    correct_conn = np.sum(within_connects1 * interacts["rna_atac"])/np.sum(within_connects1)
    false_conn = np.sum(((1 - within_connects1)) * interacts["rna_atac"])/np.sum(1 - within_connects1)
    print("ATAC_RNA")
    # Precision, true positive/(true positive + false positive)
    print("Precision: " + str(correct_conn))
    # False omission rate (FOR): false negative/(false negative + true negative)
    print("FOR: " + str(false_conn))
    print("Ratio: " + str(correct_conn/false_conn))
    print()

    correct_conn = np.sum(within_connects2 * interacts["rna_protein"])/np.sum(within_connects1)
    false_conn = np.sum(((1 - within_connects2)) * interacts["rna_protein"])/np.sum(1 - within_connects1)
    print("PROTEIN_RNA")
    print("Precision: " + str(correct_conn))
    print("FOR: " + str(false_conn))
    print("Ratio: " + str(correct_conn/false_conn))
    print()

print("sparsemax")
for alpha in ([[1000, 1, 100, 100, 0.00]]):
    print("alpha: " + str(alpha))
    model1 = model.cfrm_new3(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
    losses1 = model1.train_func(T = T)
    model1.assign_clusters(relocate_empty = True, n_relocate = None)

    # calculate the assignment accuracy 
    within_connects1 = np.zeros_like(interacts["rna_atac"])
    within_connects2 = np.zeros_like(interacts["rna_protein"])
        
    for clust in range(model1.binary_C_feats[0].shape[1]):
        clust_feats_rna = np.where(model1.binary_C_feats[0][:,clust] == True)[0]
        clust_feats_atac = np.where(model1.binary_C_feats[1][:,clust] == True)[0]
        clust_feats_protein = np.where(model1.binary_C_feats[2][:,clust] == True)[0]
        
        within_connects1[np.ix_(clust_feats_rna, clust_feats_atac)] = 1
        within_connects2[np.ix_(clust_feats_rna, clust_feats_protein)] = 1
        
    correct_conn = np.sum(within_connects1 * interacts["rna_atac"])/np.sum(within_connects1)
    false_conn = np.sum(((1 - within_connects1)) * interacts["rna_atac"])/np.sum(1 - within_connects1)
    print("ATAC_RNA")
    # Precision, true positive/(true positive + false positive)
    print("Precision: " + str(correct_conn))
    # False omission rate (FOR): false negative/(false negative + true negative)
    print("FOR: " + str(false_conn))
    print("Ratio: " + str(correct_conn/false_conn))
    print()

    correct_conn = np.sum(within_connects2 * interacts["rna_protein"])/np.sum(within_connects1)
    false_conn = np.sum(((1 - within_connects2)) * interacts["rna_protein"])/np.sum(1 - within_connects1)
    print("PROTEIN_RNA")
    print("Precision: " + str(correct_conn))
    print("FOR: " + str(false_conn))
    print("Ratio: " + str(correct_conn/false_conn))
    print()

# In[2]
# train model Check the speed
"""
#hyper parameters
Ns = [10] * 4
K = 10
N_feat = Ns[0] + 1
interval = 100
T = 10000
lr = 1e-3
run = 0
batchsize = 0.1
alpha = [1000, 1, 100, 100, 0.01]

start_time = time.time()
model1 = model.cfrm_new(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
losses1 = model1.train_func(T = T)
print("batchsize: "+ str(batchsize) + ", lr: " + str(lr) + ", time cost: "+ str(time.time() - start_time) + " sec")


Ns = [10] * 4
K = 10
N_feat = Ns[0] + 1
interval = 100
T = 10000
lr = 5e-3
run = 0
batchsize = 0.1
alpha = [1000, 1, 100, 100, 0.01]

start_time = time.time()
model1 = model.cfrm_new(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
losses2 = model1.train_func(T = T)
print("batchsize: "+ str(batchsize) + ", lr: " + str(lr) + ", time cost: "+ str(time.time() - start_time) + " sec")


Ns = [10] * 4
K = 10
N_feat = Ns[0] + 1
interval = 100
T = 10000
lr = 1e-2
run = 0
batchsize = 0.1
alpha = [1000, 1, 100, 100, 0.01]

start_time = time.time()
model1 = model.cfrm_new(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
losses3 = model1.train_func(T = T)
print("batchsize: "+ str(batchsize) + ", lr: " + str(lr) + ", time cost: "+ str(time.time() - start_time) + " sec")

fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
ax.plot(np.arange(interval, interval + T, interval), np.array(losses1), label = "1e-3")
ax.plot(np.arange(interval, interval + T, interval), np.array(losses2), label = "5e-3")
ax.plot(np.arange(interval, interval + T, interval), np.array(losses3), label = "1e-2")
ax.legend(fontsize = 15)
ax.set_xlabel("iteration", fontsize = 15)
ax.set_ylabel("loss", fontsize = 15)
fig.savefig("results_speed/Multi2_loss_" + str(batchsize) + ".pdf", bbox_inches = "tight")



Ns = [10] * 4
K = 10
N_feat = Ns[0] + 1
interval = 100
T = 10000
lr = 1e-3
run = 0
batchsize = 0.3
alpha = [1000, 1, 100, 100, 0.01]

start_time = time.time()
model1 = model.cfrm_new(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
losses1 = model1.train_func(T = T)
print("batchsize: "+ str(batchsize) + ", lr: " + str(lr) + ", time cost: "+ str(time.time() - start_time) + " sec")


Ns = [10] * 4
K = 10
N_feat = Ns[0] + 1
interval = 100
T = 10000
lr = 5e-3
run = 0
batchsize = 0.3
alpha = [1000, 1, 100, 100, 0.01]

start_time = time.time()
model1 = model.cfrm_new(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
losses2 = model1.train_func(T = T)
print("batchsize: "+ str(batchsize) + ", lr: " + str(lr) + ", time cost: "+ str(time.time() - start_time) + " sec")


Ns = [10] * 4
K = 10
N_feat = Ns[0] + 1
interval = 100
T = 10000
lr = 1e-2
run = 0
batchsize = 0.3
alpha = [1000, 1, 100, 100, 0.01]

start_time = time.time()
model1 = model.cfrm_new(counts = counts, interacts = interacts, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
losses3 = model1.train_func(T = T)
print("batchsize: "+ str(batchsize) + ", lr: " + str(lr) + ", time cost: "+ str(time.time() - start_time) + " sec")

fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
ax.plot(np.arange(interval, interval + T, interval), np.array(losses1), label = "1e-3")
ax.plot(np.arange(interval, interval + T, interval), np.array(losses2), label = "5e-3")
ax.plot(np.arange(interval, interval + T, interval), np.array(losses3), label = "1e-2")
ax.legend(fontsize = 15)
ax.set_xlabel("iteration", fontsize = 15)
ax.set_ylabel("loss", fontsize = 15)
fig.savefig("results_speed/Multi2_loss_" + str(batchsize) + ".pdf", bbox_inches = "tight")
"""
