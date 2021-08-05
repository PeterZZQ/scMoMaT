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

torch.cuda.empty_cache()

# In[2]
# train model
# last one is the one fit into the original model

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