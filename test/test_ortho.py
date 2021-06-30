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

import seaborn as sns

import diffmodel
import coupleNMF

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
load_dir = '../data/simulated/'
save_dir = './results_ortho/'
paths = ['2b5c_sigma0.1_b1_1/', '2b5c_sigma0.1_b1_2/', '2b5c_sigma0.2_b1_1/', '2b5c_sigma0.2_b1_2/', '2b5c_sigma0.3_b1_1/', '2b5c_sigma0.3_b1_2/']



alphas = [[1000, 1000, 100, 100, 1, 100, 100, 0], [1000, 1000, 100, 100, 1, 0, 0, 0]]
#hyper parameters
batchsize = 0.3
lr = 1e-3
runs = 5

N1 = 3
N2 = 3
K = 3   
# In[1]
for path in paths:

    # read in data
    counts_rna = pd.read_csv(os.path.join(load_dir + path, 'GxC1.txt'), sep = "\t", header = None).values.T
    counts_atac = pd.read_csv(os.path.join(load_dir + path, 'RxC2.txt'), sep = "\t", header = None).values.T
    A = pd.read_csv(os.path.join(load_dir + path, 'region2gene.txt'), sep = "\t", header = None).values.T

    counts_rna = np.array(counts_rna)
    counts_atac = np.array(counts_atac)
    A = np.array(A)
    counts = {"rna":[counts_rna], "atac": [counts_atac], "gact": [A]}

    label_rna = pd.read_csv(os.path.join(load_dir + path, "cell_label1.txt"), index_col=0, sep = "\t")["pop"].values.squeeze()
    label_atac = pd.read_csv(os.path.join(load_dir + path, "cell_label2.txt"), index_col=0, sep = "\t")["pop"].values.squeeze()
    
    for i, alpha in enumerate(alphas):
            
        ari_cfrm = []

        for run in range(runs):
            print("data: "+ path[:-1] + "\nrun: "+ str(run))
            print("CFRM...")
            # our model
            model1 = diffmodel.cfrm_diff(counts, N1 = N1, N2 = N2, K = K, batch_size = batchsize, interval = 100, lr = lr, alpha = alpha, seed = run, learn_gact = False).to(device)

            model1.train_func(T = 10000)

            with torch.no_grad():
                loss, *_ = model1.batch_loss('valid', alpha)
                print('Final Loss is {:.5f}'.format(loss.item()))

            # plots
            umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4) 

            z_rna = model1.softmax(model1.C_1.cpu().detach()).numpy()
            z_atac = model1.softmax(model1.C_2.cpu().detach()).numpy()

            max_rna = np.argmax(z_rna, axis = 1).squeeze()
            max_atac = np.argmax(z_atac, axis = 1).squeeze()
            z = umap_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))

            utils.plot_latent(z[:z_rna.shape[0],:], z[z_rna.shape[0]:,:], label_rna, label_atac, mode= "separate", save = save_dir + "plots/" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")
            
            utils.plot_latent(z[:z_rna.shape[0],:], z[z_rna.shape[0]:,:], max_rna, max_atac, mode= "separate", save = save_dir + "plots/predict_" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")
            
            g = sns.clustermap(z_rna, figsize = (7,7))
            g.savefig(save_dir + "plots/clustmap_rna_" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")

            g = sns.clustermap(z_atac, figsize = (7,7))
            g.savefig(save_dir + "plots/clustmap_atac_" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")
            
            # ARI score, same as what is used in Fabian benchmark code
            ari_cfrm.append(adjusted_rand_score(labels_pred = np.concatenate((max_rna, max_atac), axis = 0), labels_true = np.concatenate((label_rna, label_atac), axis = 0)))
        
        np.save( "./results/" + path[:-1] + "_" + str(i) + "_ari_cfrm.npy", np.array(ari_cfrm))    
        