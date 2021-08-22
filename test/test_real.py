# In[0]
from pickle import NONE
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

import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from itertools import product, permutations, combinations, combinations_with_replacement

from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay,roc_curve,auc,RocCurveDisplay, average_precision_score, roc_auc_score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def compute_auc(estm_adj, gt_adj, directed = False):
    """\
    Description:
    ------------
        calculate AUPRC and AUROC
    Parameters:
    ------------
        estm_adj: predict graph adjacency matrix
        gt_adj: ground truth graph adjacency matrix
        directed: the directed estimation or not
    Return:
    ------------
        prec: precision
        recall: recall
        fpr: false positive rate
        tpr: true positive rate
        AUPRC, AUROC
    """
    estm_norm_adj = np.abs(estm_adj)/np.max(np.abs(estm_adj) + 1e-12)
    
    if np.max(estm_norm_adj) == 0:
        return 0, 0, 0, 0, 0, 0
    else:
        # assert np.abs(np.max(estm_norm_adj) - 1) < 1e-4
        if directed == False:
            gt_adj = ((gt_adj + gt_adj.T) > 0).astype(np.int)
        np.fill_diagonal(gt_adj, 0)
        np.fill_diagonal(estm_norm_adj, 0)
        rows, cols = np.where(gt_adj != 0)

        fpr, tpr, thresholds = roc_curve(y_true=gt_adj.reshape(-1,), y_score=estm_norm_adj.reshape(-1,), pos_label=1)
        prec, recall, thresholds = precision_recall_curve(y_true=gt_adj.reshape(-1,), probas_pred=estm_norm_adj.reshape(-1,), pos_label=1)

        # the same
        # AUPRC = average_precision_score(gt_adj.reshape(-1,), estm_norm_adj.reshape(-1,)) 

        return prec, recall, fpr, tpr, auc(recall, prec), auc(fpr, tpr)


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
interacts["protein_atac"] = (interacts["rna_protein"].T @ interacts["rna_atac"]) > 0

# In[2] hyper parameters


alpha = [1000, 1, 100, 100, 0.00]
batchsize = 0.1
run = 0
Ns = [10] * 4
K = 10
N_feat = Ns[0]
interval = 10000
T = 4000
lr = 1e-2

# print("not using interaction")
print("using softmax")
print("expand latent dimension")
print("method 1")
for N_feat in [10, 20, 30]:
    print("number of latent feature dimension: {:d}".format(N_feat))
    # train model
    model1 = model.cfrm_new(counts = counts, interacts = None, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
    losses1 = model1.train_func(T = T)

    # assign clusters
    model1.assign_clusters(relocate_empty = True, n_relocate = None)

    # C_feats = [model1.C_feats[0].data.cpu().numpy(), model1.C_feats[1].data.cpu().numpy(), model1.C_feats[2].data.cpu().numpy()]
    C_feats = [model1.softmax(model1.C_feats[0]).data.cpu().numpy(), model1.softmax(model1.C_feats[1]).data.cpu().numpy(), model1.softmax(model1.C_feats[2]).data.cpu().numpy()]
    # Infer correlationship between modalities
    # RNA & ATAC
    RNA_ATAC = utils.infer_interaction(C_feats[0], C_feats[1])
    # RNA & Protein
    RNA_PROTEIN = utils.infer_interaction(C_feats[0], C_feats[2])
    # Protein & ATAC
    PROTEIN_ATAC = utils.infer_interaction(C_feats[2], C_feats[1])


    # AUPRC
    *_, AUPRC_rna_atac, AUROC_rna_atac = compute_auc(RNA_ATAC, interacts["rna_atac"], directed = True)
    *_, AUPRC_rna_protein, AUROC_rna_protein = compute_auc(RNA_PROTEIN, interacts["rna_protein"], directed = True)
    *_, AUPRC_protein_atac, AUROC_protein_atac = compute_auc(PROTEIN_ATAC, interacts["protein_atac"], directed = True)

    RNA_ATAC_rand = np.random.rand(RNA_ATAC.shape[0], RNA_ATAC.shape[1])
    RNA_PROTEIN_rand = np.random.rand(RNA_PROTEIN.shape[0], RNA_PROTEIN.shape[1])
    PROTEIN_ATAC_rand = np.random.rand(PROTEIN_ATAC.shape[0], PROTEIN_ATAC.shape[1])

    *_, AUPRC_rna_atac_rand, AUROC_rna_atac_rand = compute_auc(RNA_ATAC_rand, interacts["rna_atac"], directed = True)
    *_, AUPRC_rna_protein_rand, AUROC_rna_protein_rand = compute_auc(RNA_PROTEIN_rand, interacts["rna_protein"], directed = True)
    *_, AUPRC_protein_atac_rand, AUROC_protein_atac_rand = compute_auc(PROTEIN_ATAC_rand, interacts["protein_atac"], directed = True)

    AUPRC_rna_atac_ratio = AUPRC_rna_atac/AUPRC_rna_atac_rand
    AUPRC_rna_protein_ratio = AUPRC_rna_protein/AUPRC_rna_protein_rand
    AUPRC_protein_atac_ratio = AUPRC_protein_atac/AUPRC_protein_atac_rand

    print("AUPRC (protein & ATAC): {:.3f} ".format(AUPRC_protein_atac_ratio))
    print("AUPRC (RNA & ATAC): {:.3f}".format(AUPRC_rna_atac_ratio))
    print("AUPRC (RNA & protein): {:.3f}".format(AUPRC_rna_protein_ratio))



    # Average weight
    ave_rna_atac_foreground = np.sum(interacts["rna_atac"] * RNA_ATAC)/np.sum(interacts["rna_atac"])
    ave_rna_atac_background = np.sum((1 - interacts["rna_atac"]) * RNA_ATAC)/np.sum(1 - interacts["rna_atac"])

    print("average rna_atac foreground: {:.3f}".format(ave_rna_atac_foreground))
    print("average rna_atac background: {:.3f}".format(ave_rna_atac_background))

    ave_rna_protein_foreground = np.sum(interacts["rna_protein"] * RNA_PROTEIN)/np.sum(interacts["rna_protein"])
    ave_rna_protein_background = np.sum((1 - interacts["rna_protein"]) * RNA_PROTEIN)/np.sum(1 - interacts["rna_protein"])

    print("average rna_protein foreground: {:.3f}".format(ave_rna_protein_foreground))
    print("average rna_protein background: {:.3f}".format(ave_rna_protein_background))

    ave_protein_atac_foreground = np.sum(interacts["protein_atac"] * PROTEIN_ATAC)/np.sum(interacts["protein_atac"])
    ave_protein_atac_background = np.sum((1 - interacts["protein_atac"]) * PROTEIN_ATAC)/np.sum(1 - interacts["protein_atac"])

    print("average protein_atac foreground: {:.3f}".format(ave_protein_atac_foreground))
    print("average protein_atac background: {:.3f}".format(ave_protein_atac_background))


print("method 2")
for N_feat in [10, 20, 30]:
    print("number of latent feature dimension: {:d}".format(N_feat))
    # train model
    model1 = model.cfrm_new2(counts = counts, interacts = None, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
    losses1 = model1.train_func(T = T)

    # assign clusters
    model1.assign_clusters(relocate_empty = True, n_relocate = None)
    # C_feats = [model1.C_feats[0].data.cpu().numpy(), model1.C_feats[1].data.cpu().numpy(), model1.C_feats[2].data.cpu().numpy()]
    C_feats = [model1.softmax(model1.C_feats[0]).data.cpu().numpy(), model1.softmax(model1.C_feats[1]).data.cpu().numpy(), model1.softmax(model1.C_feats[2]).data.cpu().numpy()]
    # Infer correlationship between modalities
    # RNA & ATAC
    RNA_ATAC = utils.infer_interaction(C_feats[0], C_feats[1])
    # RNA & Protein
    RNA_PROTEIN = utils.infer_interaction(C_feats[0], C_feats[2])
    # Protein & ATAC
    PROTEIN_ATAC = utils.infer_interaction(C_feats[2], C_feats[1])


    # AUPRC
    *_, AUPRC_rna_atac, AUROC_rna_atac = compute_auc(RNA_ATAC, interacts["rna_atac"], directed = True)
    *_, AUPRC_rna_protein, AUROC_rna_protein = compute_auc(RNA_PROTEIN, interacts["rna_protein"], directed = True)
    *_, AUPRC_protein_atac, AUROC_protein_atac = compute_auc(PROTEIN_ATAC, interacts["protein_atac"], directed = True)

    RNA_ATAC_rand = np.random.rand(RNA_ATAC.shape[0], RNA_ATAC.shape[1])
    RNA_PROTEIN_rand = np.random.rand(RNA_PROTEIN.shape[0], RNA_PROTEIN.shape[1])
    PROTEIN_ATAC_rand = np.random.rand(PROTEIN_ATAC.shape[0], PROTEIN_ATAC.shape[1])

    *_, AUPRC_rna_atac_rand, AUROC_rna_atac_rand = compute_auc(RNA_ATAC_rand, interacts["rna_atac"], directed = True)
    *_, AUPRC_rna_protein_rand, AUROC_rna_protein_rand = compute_auc(RNA_PROTEIN_rand, interacts["rna_protein"], directed = True)
    *_, AUPRC_protein_atac_rand, AUROC_protein_atac_rand = compute_auc(PROTEIN_ATAC_rand, interacts["protein_atac"], directed = True)

    AUPRC_rna_atac_ratio = AUPRC_rna_atac/AUPRC_rna_atac_rand
    AUPRC_rna_protein_ratio = AUPRC_rna_protein/AUPRC_rna_protein_rand
    AUPRC_protein_atac_ratio = AUPRC_protein_atac/AUPRC_protein_atac_rand

    print("AUPRC (protein & ATAC): {:.3f} ".format(AUPRC_protein_atac_ratio))
    print("AUPRC (RNA & ATAC): {:.3f}".format(AUPRC_rna_atac_ratio))
    print("AUPRC (RNA & protein): {:.3f}".format(AUPRC_rna_protein_ratio))



    # Average weight
    ave_rna_atac_foreground = np.sum(interacts["rna_atac"] * RNA_ATAC)/np.sum(interacts["rna_atac"])
    ave_rna_atac_background = np.sum((1 - interacts["rna_atac"]) * RNA_ATAC)/np.sum(1 - interacts["rna_atac"])

    print("average rna_atac foreground: {:.3f}".format(ave_rna_atac_foreground))
    print("average rna_atac background: {:.3f}".format(ave_rna_atac_background))

    ave_rna_protein_foreground = np.sum(interacts["rna_protein"] * RNA_PROTEIN)/np.sum(interacts["rna_protein"])
    ave_rna_protein_background = np.sum((1 - interacts["rna_protein"]) * RNA_PROTEIN)/np.sum(1 - interacts["rna_protein"])

    print("average rna_protein foreground: {:.3f}".format(ave_rna_protein_foreground))
    print("average rna_protein background: {:.3f}".format(ave_rna_protein_background))

    ave_protein_atac_foreground = np.sum(interacts["protein_atac"] * PROTEIN_ATAC)/np.sum(interacts["protein_atac"])
    ave_protein_atac_background = np.sum((1 - interacts["protein_atac"]) * PROTEIN_ATAC)/np.sum(1 - interacts["protein_atac"])

    print("average protein_atac foreground: {:.3f}".format(ave_protein_atac_foreground))
    print("average protein_atac background: {:.3f}".format(ave_protein_atac_background))






dir = '../data/real/ASAP-PBMC/'
print("bootstrap")
print("method 1")
n_batches = 2
models = []
C_feats = [[], [], []]

alpha = [1000, 1, 100, 100, 0.00]
batchsize = 0.1
run = 0
Ns = [10] * 4
K = 10
N_feat = Ns[0]
interval = 10000
T = 4000
lr = 1e-2

for subsample in range(5):
    counts_rnas = []
    counts_atacs = []
    counts_proteins = []

    for batch in range(1,5):
        try:
            counts_atac = np.array(sp.load_npz(os.path.join(dir, 'RxC' + str(batch) + ".npz")).todense().T)
            # start_idx = subsample * int(counts_atac.shape[0]/n_batches)
            # end_idx = start_idx + int(counts_atac.shape[0]/n_batches)
            indices = np.random.choice(counts_atac.shape[0], int(counts_atac.shape[0]/n_batches), replace = False)
            counts_atac = utils.preprocess(counts_atac, modality = "ATAC")[indices, :]
        except:
            counts_atac = None
            
        try:
            counts_rna = np.array(sp.load_npz(os.path.join(dir, 'GxC' + str(batch) + ".npz")).todense().T)
            # start_idx = subsample * int(counts_rna.shape[0]/n_batches)
            # end_idx = start_idx + int(counts_rna.shape[0]/n_batches)
            indices = np.random.choice(counts_rna.shape[0], int(counts_rna.shape[0]/n_batches), replace = False)
            counts_rna = utils.preprocess(counts_rna, modality = "RNA")[indices, :]
        except:
            counts_rna = None
        
        try:
            counts_protein = np.array(sp.load_npz(os.path.join(dir, 'PxC' + str(batch) + ".npz")).todense().T)
            # start_idx = subsample * int(counts_protein.shape[0]/n_batches)
            # end_idx = start_idx + int(counts_protein.shape[0]/n_batches)
            indices = np.random.choice(counts_protein.shape[0], int(counts_protein.shape[0]/n_batches), replace = False)
            counts_protein = utils.preprocess(counts_protein, modality = "RNA")[indices, :]
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
    interacts["protein_atac"] = (interacts["rna_protein"].T @ interacts["rna_atac"]) > 0

    # train model
    model1 = model.cfrm_new(counts = counts, interacts = None, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
    losses1 = model1.train_func(T = T)

    # assign clusters
    model1.assign_clusters(relocate_empty = True, n_relocate = None)

    # concatenate all matrices
    # C_feats[0].append(model1.C_feats[0].data.cpu().numpy())
    # C_feats[1].append(model1.C_feats[1].data.cpu().numpy())
    # C_feats[2].append(model1.C_feats[2].data.cpu().numpy())
    C_feats[0].append(model1.softmax(model1.C_feats[0]).data.cpu().numpy())
    C_feats[1].append(model1.softmax(model1.C_feats[1]).data.cpu().numpy())
    C_feats[2].append(model1.softmax(model1.C_feats[2]).data.cpu().numpy())

    models.append(model1)

# concatenate all matrices
C_feats[0] = np.concatenate(C_feats[0], axis = 1)
assert C_feats[0].shape[0] == model1.C_feats[0].shape[0]
assert len(C_feats[0].shape) == 2
C_feats[1] = np.concatenate(C_feats[1], axis = 1)
assert C_feats[1].shape[0] == model1.C_feats[1].shape[0]
assert len(C_feats[1].shape) == 2
C_feats[2] = np.concatenate(C_feats[2], axis = 1)
assert C_feats[2].shape[0] == model1.C_feats[2].shape[0]
assert len(C_feats[2].shape) == 2


# Infer correlationship between modalities
# RNA & ATAC
RNA_ATAC = utils.infer_interaction(C_feats[0], C_feats[1])
# RNA & Protein
RNA_PROTEIN = utils.infer_interaction(C_feats[0], C_feats[2])
# Protein & ATAC
PROTEIN_ATAC = utils.infer_interaction(C_feats[2], C_feats[1])

# AUPRC
*_, AUPRC_rna_atac, AUROC_rna_atac = compute_auc(RNA_ATAC, interacts["rna_atac"], directed = True)
*_, AUPRC_rna_protein, AUROC_rna_protein = compute_auc(RNA_PROTEIN, interacts["rna_protein"], directed = True)
*_, AUPRC_protein_atac, AUROC_protein_atac = compute_auc(PROTEIN_ATAC, interacts["protein_atac"], directed = True)

RNA_ATAC_rand = np.random.rand(RNA_ATAC.shape[0], RNA_ATAC.shape[1])
RNA_PROTEIN_rand = np.random.rand(RNA_PROTEIN.shape[0], RNA_PROTEIN.shape[1])
PROTEIN_ATAC_rand = np.random.rand(PROTEIN_ATAC.shape[0], PROTEIN_ATAC.shape[1])

*_, AUPRC_rna_atac_rand, AUROC_rna_atac_rand = compute_auc(RNA_ATAC_rand, interacts["rna_atac"], directed = True)
*_, AUPRC_rna_protein_rand, AUROC_rna_protein_rand = compute_auc(RNA_PROTEIN_rand, interacts["rna_protein"], directed = True)
*_, AUPRC_protein_atac_rand, AUROC_protein_atac_rand = compute_auc(PROTEIN_ATAC_rand, interacts["protein_atac"], directed = True)

AUPRC_rna_atac_ratio = AUPRC_rna_atac/AUPRC_rna_atac_rand
AUPRC_rna_protein_ratio = AUPRC_rna_protein/AUPRC_rna_protein_rand
AUPRC_protein_atac_ratio = AUPRC_protein_atac/AUPRC_protein_atac_rand

print("AUPRC (protein & ATAC): {:.3f} ".format(AUPRC_protein_atac_ratio))
print("AUPRC (RNA & ATAC): {:.3f}".format(AUPRC_rna_atac_ratio))
print("AUPRC (RNA & protein): {:.3f}".format(AUPRC_rna_protein_ratio))

    
print("method 2")
C_feats = [[], [], []]

for subsample in range(5):
    counts_rnas = []
    counts_atacs = []
    counts_proteins = []

    for batch in range(1,5):
        try:
            counts_atac = np.array(sp.load_npz(os.path.join(dir, 'RxC' + str(batch) + ".npz")).todense().T)
            # start_idx = subsample * int(counts_atac.shape[0]/n_batches)
            # end_idx = start_idx + int(counts_atac.shape[0]/n_batches)
            indices = np.random.choice(counts_atac.shape[0], int(counts_atac.shape[0]/n_batches), replace = False)
            counts_atac = utils.preprocess(counts_atac, modality = "ATAC")[indices, :]
        except:
            counts_atac = None
            
        try:
            counts_rna = np.array(sp.load_npz(os.path.join(dir, 'GxC' + str(batch) + ".npz")).todense().T)
            # start_idx = subsample * int(counts_rna.shape[0]/n_batches)
            # end_idx = start_idx + int(counts_rna.shape[0]/n_batches)
            indices = np.random.choice(counts_rna.shape[0], int(counts_rna.shape[0]/n_batches), replace = False)
            counts_rna = utils.preprocess(counts_rna, modality = "RNA")[indices, :]
        except:
            counts_rna = None
        
        try:
            counts_protein = np.array(sp.load_npz(os.path.join(dir, 'PxC' + str(batch) + ".npz")).todense().T)
            # start_idx = subsample * int(counts_protein.shape[0]/n_batches)
            # end_idx = start_idx + int(counts_protein.shape[0]/n_batches)
            indices = np.random.choice(counts_protein.shape[0], int(counts_protein.shape[0]/n_batches), replace = False)
            counts_protein = utils.preprocess(counts_protein, modality = "RNA")[indices, :]
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
    interacts["protein_atac"] = (interacts["rna_protein"].T @ interacts["rna_atac"]) > 0

    # train model
    model1 = model.cfrm_new2(counts = counts, interacts = None, Ns = Ns, K = K, N_feat = N_feat, batch_size = batchsize, interval = interval, lr = lr, alpha = alpha, seed = run).to(device)
    losses1 = model1.train_func(T = T)

    # assign clusters
    model1.assign_clusters(relocate_empty = True, n_relocate = None)

    # concatenate all matrices
    # C_feats[0].append(model1.C_feats[0].data.cpu().numpy())
    # C_feats[1].append(model1.C_feats[1].data.cpu().numpy())
    # C_feats[2].append(model1.C_feats[2].data.cpu().numpy())
    C_feats[0].append(model1.softmax(model1.C_feats[0]).data.cpu().numpy())
    C_feats[1].append(model1.softmax(model1.C_feats[1]).data.cpu().numpy())
    C_feats[2].append(model1.softmax(model1.C_feats[2]).data.cpu().numpy())

    models.append(model1)

# concatenate all matrices
C_feats[0] = np.concatenate(C_feats[0], axis = 1)
assert C_feats[0].shape[0] == model1.C_feats[0].shape[0]
assert len(C_feats[0].shape) == 2
C_feats[1] = np.concatenate(C_feats[1], axis = 1)
assert C_feats[1].shape[0] == model1.C_feats[1].shape[0]
assert len(C_feats[1].shape) == 2
C_feats[2] = np.concatenate(C_feats[2], axis = 1)
assert C_feats[2].shape[0] == model1.C_feats[2].shape[0]
assert len(C_feats[2].shape) == 2


# Infer correlationship between modalities
# RNA & ATAC
RNA_ATAC = utils.infer_interaction(C_feats[0], C_feats[1])
# RNA & Protein
RNA_PROTEIN = utils.infer_interaction(C_feats[0], C_feats[2])
# Protein & ATAC
PROTEIN_ATAC = utils.infer_interaction(C_feats[2], C_feats[1])

# AUPRC
*_, AUPRC_rna_atac, AUROC_rna_atac = compute_auc(RNA_ATAC, interacts["rna_atac"], directed = True)
*_, AUPRC_rna_protein, AUROC_rna_protein = compute_auc(RNA_PROTEIN, interacts["rna_protein"], directed = True)
*_, AUPRC_protein_atac, AUROC_protein_atac = compute_auc(PROTEIN_ATAC, interacts["protein_atac"], directed = True)

RNA_ATAC_rand = np.random.rand(RNA_ATAC.shape[0], RNA_ATAC.shape[1])
RNA_PROTEIN_rand = np.random.rand(RNA_PROTEIN.shape[0], RNA_PROTEIN.shape[1])
PROTEIN_ATAC_rand = np.random.rand(PROTEIN_ATAC.shape[0], PROTEIN_ATAC.shape[1])

*_, AUPRC_rna_atac_rand, AUROC_rna_atac_rand = compute_auc(RNA_ATAC_rand, interacts["rna_atac"], directed = True)
*_, AUPRC_rna_protein_rand, AUROC_rna_protein_rand = compute_auc(RNA_PROTEIN_rand, interacts["rna_protein"], directed = True)
*_, AUPRC_protein_atac_rand, AUROC_protein_atac_rand = compute_auc(PROTEIN_ATAC_rand, interacts["protein_atac"], directed = True)

AUPRC_rna_atac_ratio = AUPRC_rna_atac/AUPRC_rna_atac_rand
AUPRC_rna_protein_ratio = AUPRC_rna_protein/AUPRC_rna_protein_rand
AUPRC_protein_atac_ratio = AUPRC_protein_atac/AUPRC_protein_atac_rand

print("AUPRC (protein & ATAC): {:.3f} ".format(AUPRC_protein_atac_ratio))
print("AUPRC (RNA & ATAC): {:.3f}".format(AUPRC_rna_atac_ratio))
print("AUPRC (RNA & protein): {:.3f}".format(AUPRC_rna_protein_ratio))

    


# %%
