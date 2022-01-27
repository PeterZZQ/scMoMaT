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

paths = ['2b5c_sigma0.1_b2_1/', '2b5c_sigma0.1_b2_2/', '2b5c_sigma0.2_b2_1/', '2b5c_sigma0.2_b2_2/', '2b5c_sigma0.3_b2_1/', '2b5c_sigma0.3_b2_2/']
paths = ['2b5c_sigma0.1_b1_1/', '2b5c_sigma0.1_b1_2/', '2b5c_sigma0.2_b1_1/', '2b5c_sigma0.2_b1_2/', '2b5c_sigma0.3_b1_1/', '2b5c_sigma0.3_b1_2/']
paths = ['gene_sigma_0.1_1/', 'gene_sigma_0.1_2/', 'gene_sigma_0.2_1/', 'gene_sigma_0.2_2/', 'gene_sigma_0.3_1/', 'gene_sigma_0.3_2/', 'gene_sigma_0.4_1/', 'gene_sigma_0.4_2/', 'gene_sigma_0.5_1/', 'gene_sigma_0.5_2/']

paths = ['gene_sigma_0.5_1/', 'gene_sigma_0.5_2/']



# last one is the one fit into the original model
alphas = [[1000, 1000, 100, 100, 1, 1, 1, 0, 0.1], [1000, 1000, 100, 100, 1, 0, 1, 0, 0.1], [1000, 1000, 100, 100, 1, 0, 0, 0, 0.1]]
#hyper parameters
batchsize = 0.3
lr = 1e-3
runs = 5

Ns = [3, 3]
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

    label_g = pd.read_csv(os.path.join(dir + path, "gene_label.txt"), sep = "\t")["x"].values.squeeze()
    # calculate label_r using gene activity matrix
    label_r = []
    for region in range(A.shape[1]):
        genes = np.where(A[:,region] != 0)[0]
        glabel = label_g[genes]
        unique_gene, unique_counts = np.unique(glabel, return_counts = True)
        if len(unique_gene) !=0:
            label_r.append(unique_gene[np.argmax(unique_counts)])
        else:
            # not belonging to any clusters
            label_r.append(0)
    label_r = np.array(label_r)

    for i, alpha in enumerate(alphas):
            
        ari_gene = []
        ari_region = []
        losses = []
        #######################################################################################################
        
        # Vertical

        #######################################################################################################
        for run in range(runs):
            print("data: "+ path[:-1] + "\nrun: "+ str(run))
            print("CFRM...")
            # our model
            model1 = model.cfrm(counts1, Ns = Ns[0:1], K = K, N_feat = N_feat, batch_size = batchsize, interval=1000, lr=1e-3, alpha =alpha, seed = run).to(device)

            model1.train_func(T = 10000)

            with torch.no_grad():
                loss, *_ = model1.batch_loss('valid', alpha)
                print('Final Loss is {:.5f}'.format(loss.item()))
            
            losses.append(loss.item())
            np.save("./results_gclust/vert/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_C1.npy", model1.Cs[0].cpu().detach().numpy())
            np.save("./results_gclust/vert/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Ag.npy", model1.Ag.cpu().detach().numpy())
            np.save("./results_gclust/vert/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Ar.npy", model1.Ar.cpu().detach().numpy())
            np.save("./results_gclust/vert/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Cg.npy", model1.Cg.cpu().detach().numpy())
            np.save("./results_gclust/vert/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Cr.npy", model1.Cr.cpu().detach().numpy())
            # plots
            umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4) 

            z_g = model1.softmax(model1.Cg.cpu().detach()).numpy()
            z_r = model1.softmax(model1.Cr.cpu().detach()).numpy()

            max_g = np.argmax(z_g, axis = 1).squeeze()
            max_r = np.argmax(z_r, axis = 1).squeeze()
            z_g = umap_op.fit_transform(z_g) 
            z_r = umap_op.fit_transform(z_r)

            utils.plot_latent(z_g, z_r, label_g, label_r, mode= "separate", save = "./results_gclust/vert/plots/" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")            
            utils.plot_latent(z_g, z_r, max_g, max_r, mode= "separate", save = "./results_gclust/vert/plots/predict_" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")
            
            
            # ARI score, same as what is used in Fabian benchmark code
            ari_gene.append(adjusted_rand_score(labels_pred = max_g, labels_true = label_g))
            ari_region.append(adjusted_rand_score(labels_pred = max_r, labels_true = label_r))
        
        np.save( "./results_gclust/vert/" + path[:-1] + "_" + str(i) + "_ari_gene.npy", np.array(ari_gene)) 
        np.save( "./results_gclust/vert/" + path[:-1] + "_" + str(i) + "_ari_region.npy", np.array(ari_region)) 
        np.save( "./results_gclust/vert/" + path[:-1] + "_" + str(i) + "_losses.npy", np.array(losses))
   

    
    
    for i, alpha in enumerate(alphas):
        
        ari_gene = []
        losses = []
        #######################################################################################################
        
        # Horizontal

        #######################################################################################################
        for run in range(runs):
            print("data: "+ path[:-1] + "\nrun: "+ str(run))
            print("CFRM...")
            # our model
            model1 = model.cfrm(counts2, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval=1000, lr=1e-3, alpha =alpha, seed = run).to(device)

            model1.train_func(T = 10000)

            with torch.no_grad():
                loss, *_ = model1.batch_loss('valid', alpha)
                print('Final Loss is {:.5f}'.format(loss.item()))
            
            losses.append(loss.item())
            np.save("./results_gclust/hori/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_C1.npy", model1.Cs[0].cpu().detach().numpy())
            np.save("./results_gclust/hori/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_C2.npy", model1.Cs[1].cpu().detach().numpy())
            np.save("./results_gclust/hori/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Ag.npy", model1.Ag.cpu().detach().numpy())
            np.save("./results_gclust/hori/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Cg.npy", model1.Cg.cpu().detach().numpy())
            
            # plots
            umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4) 

            z_g = model1.softmax(model1.Cg.cpu().detach()).numpy()

            max_g = np.argmax(z_g, axis = 1).squeeze()
            z = umap_op.fit_transform(z_g)

            utils.plot_latent(z, z, label_g, label_g, mode= "separate", save = "./results_gclust/hori/plots/" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")
            
            utils.plot_latent(z, z, max_g, max_g, mode= "separate", save = "./results_gclust/hori/plots/predict_" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")
            
            
            # ARI score, same as what is used in Fabian benchmark code
            ari_gene.append(adjusted_rand_score(labels_pred = max_g, labels_true = label_g))
        
        np.save( "./results_gclust/hori/" + path[:-1] + "_" + str(i) + "_ari_gene.npy", np.array(ari_gene)) 
        np.save( "./results_gclust/hori/" + path[:-1] + "_" + str(i) + "_losses.npy", np.array(losses))
   

    
    
    for i, alpha in enumerate(alphas):
          
        ari_gene = []
        ari_region = []
        losses = []
        #######################################################################################################
        
        # Diagonal

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
            np.save("./results_gclust/diag/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_C1.npy", model1.Cs[0].cpu().detach().numpy())
            np.save("./results_gclust/diag/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_C2.npy", model1.Cs[1].cpu().detach().numpy())
            np.save("./results_gclust/diag/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Ag.npy", model1.Ag.cpu().detach().numpy())
            np.save("./results_gclust/diag/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Ar.npy", model1.Ar.cpu().detach().numpy())
            np.save("./results_gclust/diag/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Cg.npy", model1.Cg.cpu().detach().numpy())
            np.save("./results_gclust/diag/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Cr.npy", model1.Cr.cpu().detach().numpy())
            # plots
            umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4) 

            z_g = model1.softmax(model1.Cg.cpu().detach()).numpy()
            z_r = model1.softmax(model1.Cr.cpu().detach()).numpy()

            max_g = np.argmax(z_g, axis = 1).squeeze()
            max_r = np.argmax(z_r, axis = 1).squeeze()
            z_g = umap_op.fit_transform(z_g) 
            z_r = umap_op.fit_transform(z_r)

            utils.plot_latent(z_g, z_r, label_g, label_r, mode= "separate", save = "./results_gclust/diag/plots/" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")            
            utils.plot_latent(z_g, z_r, max_g, max_r, mode= "separate", save = "./results_gclust/diag/plots/predict_" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")
            
            
            # ARI score, same as what is used in Fabian benchmark code
            ari_gene.append(adjusted_rand_score(labels_pred = max_g, labels_true = label_g))
            ari_region.append(adjusted_rand_score(labels_pred = max_r, labels_true = label_r))
        
        np.save( "./results_gclust/diag/" + path[:-1] + "_" + str(i) + "_ari_gene.npy", np.array(ari_gene)) 
        np.save( "./results_gclust/diag/" + path[:-1] + "_" + str(i) + "_ari_region.npy", np.array(ari_region)) 
        np.save( "./results_gclust/diag/" + path[:-1] + "_" + str(i) + "_losses.npy", np.array(losses))
   

    for i, alpha in enumerate(alphas):
           
        ari_gene = []
        ari_region = []
        losses = []
        #######################################################################################################
        
        # Multi3

        #######################################################################################################
        for run in range(runs):
            print("data: "+ path[:-1] + "\nrun: "+ str(run))
            print("CFRM...")
            # our model
            model1 = model.cfrm(counts4, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval=1000, lr=1e-3, alpha =alpha, seed = run).to(device)

            model1.train_func(T = 10000)

            with torch.no_grad():
                loss, *_ = model1.batch_loss('valid', alpha)
                print('Final Loss is {:.5f}'.format(loss.item()))
            
            losses.append(loss.item())
            
            np.save("./results_gclust/multi3/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_C1.npy", model1.Cs[0].cpu().detach().numpy())
            np.save("./results_gclust/multi3/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_C2.npy", model1.Cs[1].cpu().detach().numpy())
            np.save("./results_gclust/multi3/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Ag.npy", model1.Ag.cpu().detach().numpy())
            np.save("./results_gclust/multi3/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Ar.npy", model1.Ar.cpu().detach().numpy())
            np.save("./results_gclust/multi3/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Cg.npy", model1.Cg.cpu().detach().numpy())
            np.save("./results_gclust/multi3/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Cr.npy", model1.Cr.cpu().detach().numpy())
            
            # plots
            umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4) 

            z_g = model1.softmax(model1.Cg.cpu().detach()).numpy()
            z_r = model1.softmax(model1.Cr.cpu().detach()).numpy()

            max_g = np.argmax(z_g, axis = 1).squeeze()
            max_r = np.argmax(z_r, axis = 1).squeeze()
            z_g = umap_op.fit_transform(z_g) 
            z_r = umap_op.fit_transform(z_r)

            utils.plot_latent(z_g, z_r, label_g, label_r, mode= "separate", save = "./results_gclust/multi3/plots/" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")            
            utils.plot_latent(z_g, z_r, max_g, max_r, mode= "separate", save = "./results_gclust/multi3/plots/predict_" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")
            
  
            
            # ARI score, same as what is used in Fabian benchmark code
            ari_gene.append(adjusted_rand_score(labels_pred = max_g, labels_true = label_g))
            ari_region.append(adjusted_rand_score(labels_pred = max_r, labels_true = label_r))
        
        np.save( "./results_gclust/multi3/" + path[:-1] + "_" + str(i) + "_ari_gene.npy", np.array(ari_gene)) 
        np.save( "./results_gclust/multi3/" + path[:-1] + "_" + str(i) + "_ari_region.npy", np.array(ari_region)) 
        np.save( "./results_gclust/multi3/" + path[:-1] + "_" + str(i) + "_losses.npy", np.array(losses))
   

    for i, alpha in enumerate(alphas):
           
        ari_gene = []
        ari_region = []
        losses = []
        #######################################################################################################
        
        # Multi4

        #######################################################################################################
        for run in range(runs):
            print("data: "+ path[:-1] + "\nrun: "+ str(run))
            print("CFRM...")
            # our model
            model1 = model.cfrm(counts5, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval=1000, lr=1e-3, alpha =alpha, seed = run).to(device)

            model1.train_func(T = 10000)

            with torch.no_grad():
                loss, *_ = model1.batch_loss('valid', alpha)
                print('Final Loss is {:.5f}'.format(loss.item()))
            
            losses.append(loss.item())
            
            np.save("./results_gclust/multi4/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_C1.npy", model1.Cs[0].cpu().detach().numpy())
            np.save("./results_gclust/multi4/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_C2.npy", model1.Cs[1].cpu().detach().numpy())
            np.save("./results_gclust/multi4/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Ag.npy", model1.Ag.cpu().detach().numpy())
            np.save("./results_gclust/multi4/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Ar.npy", model1.Ar.cpu().detach().numpy())
            np.save("./results_gclust/multi4/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Cg.npy", model1.Cg.cpu().detach().numpy())
            np.save("./results_gclust/multi4/" + path[:-1] + "_" + str(i) + "_" + str(run) + "_Cr.npy", model1.Cr.cpu().detach().numpy())
            # plots
            umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.4) 

            z_g = model1.softmax(model1.Cg.cpu().detach()).numpy()
            z_r = model1.softmax(model1.Cr.cpu().detach()).numpy()

            max_g = np.argmax(z_g, axis = 1).squeeze()
            max_r = np.argmax(z_r, axis = 1).squeeze()
            z_g = umap_op.fit_transform(z_g) 
            z_r = umap_op.fit_transform(z_r)

            utils.plot_latent(z_g, z_r, label_g, label_r, mode= "separate", save = "./results_gclust/multi4/plots/" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")            
            utils.plot_latent(z_g, z_r, max_g, max_r, mode= "separate", save = "./results_gclust/multi4/plots/predict_" + path[:-1] + "_" + str(run) + "_" + str(i) + "_cfrm.png")
            
  
            
            # ARI score, same as what is used in Fabian benchmark code
            ari_gene.append(adjusted_rand_score(labels_pred = max_g, labels_true = label_g))
            ari_region.append(adjusted_rand_score(labels_pred = max_r, labels_true = label_r))
        
        np.save( "./results_gclust/multi4/" + path[:-1] + "_" + str(i) + "_ari_gene.npy", np.array(ari_gene)) 
        np.save( "./results_gclust/multi4/" + path[:-1] + "_" + str(i) + "_ari_region.npy", np.array(ari_region)) 
        np.save( "./results_gclust/multi4/" + path[:-1] + "_" + str(i) + "_losses.npy", np.array(losses))
   




    
    
    
    