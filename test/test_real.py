import sys, os
sys.path.append('../')
sys.path.append('../src/')

import torch
import numpy as np
import utils
from torch.nn import Module, Parameter
import torch.optim as opt
from utils import preprocess
import torch.nn.functional as F

import torch.optim as opt
from torch import softmax, log_softmax, Tensor
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds
import seaborn as sns

from sklearn.metrics import adjusted_rand_score


from sklearn.decomposition import PCA
from umap import UMAP

import pandas as pd 
import numpy as np 
import scipy.sparse as sp
import torch
import model
import newmodel
import diffmodel
import coupleNMF as coupleNMF

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dir = '../data/real/Xichen/small_ver'

counts_rna = sp.load_npz(os.path.join(dir, 'GxC1_small.npz'))
counts_atac = sp.load_npz(os.path.join(dir, 'RxC2_small.npz')).astype(np.float32)
A = sp.load_npz(os.path.join(dir, 'GxR_small.npz'))
subsample = 1

counts_rna = np.array(counts_rna.todense().T)
counts_atac = np.array(counts_atac.todense().T)
counts_rna = counts_rna[::subsample,:]
counts_atac = counts_atac[::subsample,:]
A = np.array(A.todense())

counts = {"rna":[counts_rna], "atac": [counts_atac], "gact": [A]}


# train model
alphas = [[10000, 1000, 100, 100, 1, 0.5, 0.5, 0, 1], [10000, 1000, 100, 100, 1, 0.5, 0, 0, 1], [1000, 1000, 100, 100, 1, 0, 0, 0, 1]]
Ns = [8, 10, 13]
batch_size = 0.1
runs = 10
                            
                            
for i, alpha in enumerate(alphas):
    for N in Ns:
        ari_cfrm = []
        for run in range(runs):
            model1 = diffmodel.cfrm_diff(counts, N1 = N, N2 = N, K = N, batch_size = batch_size, interval=1000, lr=1e-3, alpha = alpha, seed = run, learn_gact = False).to(device)
            model1.train_func(T = 10000)

            z_rna = model1.softmax(model1.C_1).cpu().detach()
            z_atac = model1.softmax(model1.C_2).cpu().detach()

            max_rna = np.argmax(z_rna.numpy(), axis = 1).squeeze()
            max_atac = np.argmax(z_atac.numpy(), axis = 1).squeeze()

            umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4) 
            z = umap_op.fit_transform(np.concatenate((z_rna.numpy(), z_atac.numpy()), axis = 0))
            label_rna = pd.read_csv("../data/real/Xichen/meta_rna.csv", index_col=0)["cell_type"].values.squeeze()
            label_atac = pd.read_csv("../data/real/Xichen/meta_atac.csv", index_col=0)["cell_type"].values.squeeze()
            
            utils.plot_latent(z[:z_rna.shape[0],:], z[z_rna.shape[0]:,:], label_rna, label_atac, mode= "separate", axis_label = "Umap", save = "./results_real/plots1/xichen_" + str(i) + "_" + str(N) + "_" + str(run) + "_sep.png")
            utils.plot_latent(z[:z_rna.shape[0],:], z[z_rna.shape[0]:,:], label_rna, label_atac, mode= "hybrid", axis_label = "Umap", save = "./results_real/plots1/xichen_" + str(i) + "_" + str(N) + "_" + str(run) + "_hybrid.png")
            
#             z_rna, z_atac = utils.match_alignment(z_rna, z_atac, k = 10)
            
#             utils.plot_latent(z_rna, z_atac, label_rna, label_atac, mode= "separate", axis_label = "Umap", save = "./results_real/plots/xichen_" + str(i) + "_" + str(N) + "_" + str(run) + "_post_sep.png")
#             utils.plot_latent(z_rna, z_atac, label_rna, label_atac, mode= "hybrid", axis_label = "Umap", save = "./results_real/plots/xichen_" + str(i) + "_" + str(N) + "_" + str(run) + "_post_hybrid.png")
            
            ari_cfrm.append(adjusted_rand_score(labels_pred = np.concatenate((max_rna, max_atac), axis = 0), labels_true = np.concatenate((label_rna, label_atac), axis = 0)))
            
        np.save( "./results_real/xichen_" + str(i) + "_" + str(N) + "_ari_cfrm1.npy", np.array(ari_cfrm))    
            
            

            
for i, alpha in enumerate(alphas):
    for N in Ns:
        ari_cfrm = []
        for run in range(runs):
            model1 = diffmodel.cfrm_diff2(counts, N1 = N, N2 = N, K = N, batch_size = batch_size, interval=1000, lr=1e-3, alpha = alpha, seed = run, learn_gact = False).to(device)
            model1.train_func(T = 10000)

            z_rna = model1.softmax(model1.C_1).cpu().detach()
            z_atac = model1.softmax(model1.C_2).cpu().detach()

            max_rna = np.argmax(z_rna.numpy(), axis = 1).squeeze()
            max_atac = np.argmax(z_atac.numpy(), axis = 1).squeeze()

            umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4) 
            z = umap_op.fit_transform(np.concatenate((z_rna.numpy(), z_atac.numpy()), axis = 0))
            label_rna = pd.read_csv("../data/real/Xichen/meta_rna.csv", index_col=0)["cell_type"].values.squeeze()
            label_atac = pd.read_csv("../data/real/Xichen/meta_atac.csv", index_col=0)["cell_type"].values.squeeze()
            
            utils.plot_latent(z[:z_rna.shape[0],:], z[z_rna.shape[0]:,:], label_rna, label_atac, mode= "separate", axis_label = "Umap", save = "./results_real/plots2/xichen_" + str(i) + "_" + str(N) + "_" + str(run) + "_sep.png")
            utils.plot_latent(z[:z_rna.shape[0],:], z[z_rna.shape[0]:,:], label_rna, label_atac, mode= "hybrid", axis_label = "Umap", save = "./results_real/plots2/xichen_" + str(i) + "_" + str(N) + "_" + str(run) + "_hybrid.png")
            
#             z_rna, z_atac = utils.match_alignment(z_rna, z_atac, k = 10)
            
#             utils.plot_latent(z_rna, z_atac, label_rna, label_atac, mode= "separate", axis_label = "Umap", save = "./results_real/plots/xichen_" + str(i) + "_" + str(N) + "_" + str(run) + "_post_sep.png")
#             utils.plot_latent(z_rna, z_atac, label_rna, label_atac, mode= "hybrid", axis_label = "Umap", save = "./results_real/plots/xichen_" + str(i) + "_" + str(N) + "_" + str(run) + "_post_hybrid.png")
            
            ari_cfrm.append(adjusted_rand_score(labels_pred = np.concatenate((max_rna, max_atac), axis = 0), labels_true = np.concatenate((label_rna, label_atac), axis = 0)))
            
        np.save( "./results_real/xichen_" + str(i) + "_" + str(N) + "_ari_cfrm2.npy", np.array(ari_cfrm))    

