import sys, os
sys.path.append('../')
sys.path.append('../src/')

import torch
import numpy as np
from torch.nn import Module, Parameter
from utils import preprocess
import torch.nn.functional as F

from sklearn.decomposition import PCA
from umap import UMAP

import pandas as pd 
import numpy as np 
import scipy.sparse as sp

from sklearn.metrics import adjusted_rand_score
import seaborn as sns
import matplotlib.pyplot as plt

import diffmodel
import coupleNMF
import quantile

import model
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings("ignore")


#######################################################################

# Test on simulated dataset

#######################################################################


def batch_mixing_entropy(C1, C2, runs = 1):
    # construct knn graph
    k = 100
    
    C = torch.cat((C1, C2), dim = 0)
    dist = utils._pairwise_distances(C, C).numpy()
    knn_index = np.argpartition(dist, kth = k - 1, axis = 1)[:,(k-1)]
    kth_dist = np.take_along_axis(dist, knn_index[:,None], axis = 1)
    knn = ((dist - kth_dist[:, None]) <= 0)


    # select random anchors, and calculate the entropy
    entropys = []
    for run in range(runs):
        random_anchor = np.random.choice(C1.shape[0] + C2.shape[0], 100, replace=False)
        p1 = np.sum(knn[random_anchor,:C1.shape[0]], axis = 1)/100
        p2 = np.sum(knn[random_anchor,C1.shape[0]:], axis = 1)/100
        entropys.append(np.sum(p1 * np.log(p1 + 1e-6)) + np.sum(p2 * np.log(p2 + 1e-6)))
    return np.array(entropys)
    

# Read in the data: 2 batches, 3 clusters
dir = '../data/simulated/'

paths = ['2b3c_sigma0.1_b2_1/', '2b3c_sigma0.1_b2_2/', '2b3c_sigma0.2_b2_1/', '2b3c_sigma0.2_b2_2/', '2b3c_sigma0.3_b2_1/', '2b3c_sigma0.3_b2_2/']
paths = ['2b3c_sigma0.1_b1_1/', '2b3c_sigma0.1_b1_2/', '2b3c_sigma0.2_b1_1/', '2b3c_sigma0.2_b1_2/', '2b3c_sigma0.3_b1_1/', '2b3c_sigma0.3_b1_2/']
paths = ['2b3c_sigma0.4_b1_1/', '2b3c_sigma0.4_b1_2/']

# last one is the one fit into the original model
alphas = [[1000, 1000, 100, 100, 1, 1, 1, 0, 0.1], [1000, 1000, 100, 100, 1, 1, 0, 0, 0.1], [1000, 1000, 100, 100, 1, 0, 0, 0, 0.1]]
#hyper parameters
batchsize = 0.3
lr = 1e-3
runs = 5

Ns = [3,3]
K = 3
N_feat = Ns[0] + 1

# In[1]
for path in paths:

    # read in data
    counts_rna1 = pd.read_csv(os.path.join(dir + path, 'GxC1.txt'), sep = "\t", header = None).values.T
    counts_rna2 = pd.read_csv(os.path.join(dir + path, 'GxC2.txt'), sep = "\t", header = None).values.T
    counts_atac1 = pd.read_csv(os.path.join(dir + path, 'RxC1.txt'), sep = "\t", header = None).values.T
    counts_atac2 = pd.read_csv(os.path.join(dir + path, 'RxC2.txt'), sep = "\t", header = None).values.T
    A = pd.read_csv(os.path.join(dir + path, 'region2gene.txt'), sep = "\t", header = None).values.T

    counts_rna1 = np.array(counts_rna1)
    counts_rna2 = np.array(counts_rna2)
    counts_atac1 = np.array(counts_atac1)
    counts_atac2 = np.array(counts_atac2)
    A = np.array(A)


    # vertical integration
    counts1 = {"rna":[counts_rna1], "atac": [counts_atac1], "gact": A}
    # horizontal integration
    counts2 = {"rna":[counts_rna1, counts_rna2], "atac": [None, None], "gact": A}
    # diagonal integration
    counts3 = {"rna":[counts_rna1, None], "atac": [None, counts_atac2], "gact": A}
    # multi-omics
    counts4 = {"rna":[counts_rna1, None], "atac": [counts_atac1, counts_atac2], "gact": A}
    counts5 = {"rna":[counts_rna1, counts_rna2], "atac": [counts_atac1, counts_atac2], "gact": A}

    label_rna = pd.read_csv(os.path.join(dir + path, "cell_label1.txt"), index_col=0, sep = "\t")["pop"].values.squeeze()
    label_atac = pd.read_csv(os.path.join(dir + path, "cell_label2.txt"), index_col=0, sep = "\t")["pop"].values.squeeze()
    """
    for i, alpha in enumerate(alphas):
            
        ari_cfrm = []
        losses = []
        #######################################################################################################
        
        # CFRM 2

        #######################################################################################################

        for run in range(runs):
            print("data: "+ path[:-1] + "\nrun: "+ str(run))
            print("CFRM...")
            # our model
            model1 = model.cfrm(counts3, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval=1000, lr=1e-3, alpha =alpha, seed = run).to(device)

            model1.train_func(T = 10000)

            with torch.no_grad():
                loss, *_ = model1.batch_loss('valid', alpha)
                print('Final Loss is {:.5f}'.format(loss.item()))

            losses.append(loss.item())
            
            np.save("./results_multi/diag/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_C0.npy", model1.Cs[0].cpu().detach().numpy())
            np.save("./results_multi/diag/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_C1.npy", model1.Cs[1].cpu().detach().numpy())
            
            # plots
            umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4) 

            z_rna = model1.softmax(model1.Cs[0].cpu().detach()).numpy()
            z_atac = model1.softmax(model1.Cs[1].cpu().detach()).numpy()

            max_rna = np.argmax(z_rna, axis = 1).squeeze()
            max_atac = np.argmax(z_atac, axis = 1).squeeze()
            z = umap_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))

            utils.plot_latent(z[:z_rna.shape[0],:], z[z_rna.shape[0]:,:], label_rna, label_atac, mode= "separate", save = "./results_multi/diag/plots/" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")
            
            utils.plot_latent(z[:z_rna.shape[0],:], z[z_rna.shape[0]:,:], max_rna, max_atac, mode= "separate", save = "./results_multi/diag/plots/predict_" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")
            
#             g = sns.clustermap(z_rna, figsize = (7,7))
#             g.savefig("./results/plots/clustmap_rna_" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")

#             g = sns.clustermap(z_atac, figsize = (7,7))
#             g.savefig("./results/plots/clustmap_atac_" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")
            
            # ARI score, same as what is used in Fabian benchmark code
            ari_cfrm.append(adjusted_rand_score(labels_pred = np.concatenate((max_rna, max_atac), axis = 0), labels_true = np.concatenate((label_rna, label_atac), axis = 0)))
        
        np.save( "./results_multi/diag/" + path[:-1] + "_" + str(i) + "_ari_cfrm.npy", np.array(ari_cfrm))  
        np.save( "./results_multi/diag/" + path[:-1] + "_" + str(i) + "_losses.npy", np.array(losses))
        
    """
    #######################################################################################################
    
    # CoupleNMF

    #######################################################################################################
    print("coupleNMF...")
    cnmf = coupleNMF.coupleNMF({"rna":[counts_rna1.T], "atac": [counts_atac2.T], "gact": [A]}, N = Ns[0], lambda1 = None, lambda2 = None)

    E_symbol = np.array([str(x) for x in range(counts_rna1.shape[0])])
    P_symbol = np.array([str(x) for x in range(counts_atac2.shape[0])])

    H1, H2 = cnmf.train_func(E_symbol = E_symbol, P_symbol = P_symbol)

    
    umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4) 

    z_rna = H2.T
    z_atac = H1.T
    
    # np.save("./results_multi/diag/" + path[:-1] + "_C0.npy", z_rna)
    # np.save("./results_multi/diag/" + path[:-1] + "_C1.npy", z_atac)
    
    max_rna = np.argmax(z_rna, axis = 1)
    max_atac = np.argmax(z_atac, axis = 1)
    z = umap_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))

    utils.plot_latent(z[:z_rna.shape[0],:], z[z_rna.shape[0]:,:], label_rna, label_atac, mode= "separate", save = "./results_multi/diag/plots/" + path[:-1] + "_coupleNMF.png", axis_label = "Umap")
    utils.plot_latent(z[:z_rna.shape[0],:], z[z_rna.shape[0]:,:], label_rna, label_atac, mode= "separate", save = "./results_multi/diag/plots/predict_" + path[:-1] + "_coupleNMF.png", axis_label = "Umap")
    
#     g = sns.clustermap(z_rna, figsize = (7,7))
#     g.savefig("./results_multi/diag/plots/clustmap_rna_" + path[:-1] + "_coupleNMF.png")

#     g = sns.clustermap(z_atac, figsize = (7,7))
#     g.savefig("./results_multi/diag/plots/clustmap_atac_" + path[:-1] + "_coupleNMF.png")
        
    ari_coupleNMF = [adjusted_rand_score(labels_pred = np.concatenate((max_rna, max_atac), axis = 0), labels_true = np.concatenate((label_rna, label_atac), axis = 0))] * runs 
    
    np.save( "./results_multi/diag/" + path[:-1] + "_ari_coupleNMF.npy", np.array(ari_coupleNMF))
    
    z_rna, z_atac = quantile.quantile_norm([z_rna, z_atac], min_cells = 1, max_sample = 2000, quantiles = 50, ref = None, refine = True)
    max_rna = np.argmax(z_rna, axis = 1)
    max_atac = np.argmax(z_atac, axis = 1)
    ari_coupleNMF = [adjusted_rand_score(labels_pred = np.concatenate((max_rna, max_atac), axis = 0), labels_true = np.concatenate((label_rna, label_atac), axis = 0))] * runs 
    np.save( "./results_multi/diag/" + path[:-1] + "_ari_coupleNMF_quantile.npy", np.array(ari_coupleNMF))

    
    