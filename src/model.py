import sys, os
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as opt
import pandas as pd
from torch.nn import Module, Parameter, Embedding, ParameterList
from torch import softmax, log_softmax, Tensor
import itertools

from sklearn.cluster import KMeans
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import time
import utils

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class cfrm_new(Module):
    """\
        Gene clusters more than cell clusters, force A_r and A_g to be sparse:
        
        alpha[0]: the weight of the tri-factorization term
        alpha[1]: the weight of the missing dimension term
        alpha[2]: the weight of the association relationship between modalities
        alpha[3]: the weight of the interaction matrix
        alpha[4]: the sparsity of A_r and A_g
        
    """
    def __init__(self, counts, interacts, Ns, K, N_feat = None, batch_size=0.3, interval=10, lr=1e-3, alpha = [1000, 0, 100, 100, 0.1], seed = None):
        super().__init__()
        
        # init parameters, Ns is a list with length the number of batches
        self.Ns = Ns
        self.K = K
        self.N_cell = sum(self.Ns) - (len(self.Ns) - 1) * self.K
        if N_feat is None:
            self.N_feat = self.N_cell + 1
        else:
            self.N_feat = N_feat
        
        self.batch_size = batch_size
        self.interval = interval
        self.alpha = torch.FloatTensor(alpha).to(device)
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 1. load count matrices
        self.mods = [mod for mod in counts.keys()]
        self.Xs = {}
        # mod include: RNA, ATAC, PROTEIN, etc.
        for mod in self.mods:
            self.Xs[mod] = []
            for counts_mod in counts[mod]:
                if counts_mod is not None:
                    self.Xs[mod].append(torch.FloatTensor(counts_mod).to(device))
                else:
                    self.Xs[mod].append(None)
        
        self.As = {}
        if interacts is not None:
            # mods include:  RNA_ATAC, RNA_PROTEIN, etc.
            for mods in interacts.keys():
                self.As[mods] = torch.FloatTensor(interacts[mods]).to(device)
        
        # put into sanity check
        self.sanity_check()
        
        
        # 2. create parameters
        self.C_cells = ParameterList([])
        self.C_feats = ParameterList([])
        self.A_assos = ParameterList([]) 
        self.b_cells = {}
        self.b_feats = {}
        
        
        # create C_cells
        for batch in range(len(self.Ns)):
            for mod in self.mods:
                if self.Xs[mod][batch] is not None:
                    self.C_cells.append(Parameter(torch.rand(self.Xs[mod][batch].shape[0], self.N_cell)))
                    break
        
        # create C_feats, A_assos
        for mod in self.mods:
            for batch in range(len(self.Ns)):
                if self.Xs[mod][batch] is not None:
                    self.C_feats.append(Parameter(torch.rand(self.Xs[mod][batch].shape[1], self.N_feat)))
                    self.A_assos.append(Parameter(torch.rand(self.N_cell, self.N_feat)))
                    break
        
        # create bias term
        for mod in self.mods:
            self.b_cells[mod] = []
            self.b_feats[mod] = []
            for batch in range(len(self.Ns)):
                if self.Xs[mod][batch] is not None:
                    self.b_cells[mod].append(torch.zeros(self.Xs[mod][batch].shape[0], 1).to(device))
                    self.b_feats[mod].append(torch.zeros(1, self.Xs[mod][batch].shape[1]).to(device))
                else:
                    self.b_cells[mod].append(None)
                    self.b_feats[mod].append(None)


        
        # missing masks
        self.dims = []
        self.missing_dims = []
        self.mask_diags = []
        K_comps = [N - self.K for N in self.Ns]
        point = 0
        for K_comp in K_comps:
            self.dims.append([x for x in range(self.K)] + [x for x in range(self.K + point, self.K + point + K_comp)])
            self.missing_dims.append([x for x in range(self.K, self.K + point)] + [x for x in range(self.K + point + K_comp, self.N_cell)])
            self.mask_diags.append(torch.eye(len(self.dims[-1])).to(device))
            point += K_comp
        
        self.optimizer = opt.Adam(self.parameters(), lr=lr)
        
    
    def sanity_check(self):
        print("Input sanity check...")
        # number of batches are the same
        n_features = {}
        for mod in self.Xs.keys():
            # No all None modality
            if np.all(np.array([x is None for x in self.Xs[mod]]) == True) == True:
                raise ValueError("Don't have count matrix correspond to " + mod)
            
            if (len(self.Xs[mod]) == len(self.Ns)) == False:
                raise ValueError("Number of batches not match for " + mod)
            
            # feature dimension should be the same
            n_features[mod] = [x.shape[1] for x in self.Xs[mod] if x is not None]
            if np.all(np.array(n_features[mod]) == n_features[mod][0]) == False:
                raise ValueError("Number of features not match for modality " + mod)
        

        for batch in range(len(self.Ns)):
            # cell number of each batch should be the same
            n_cells = np.array([self.Xs[mod][batch].shape[0] for mod in self.Xs.keys() if self.Xs[mod][batch] is not None])
            if len(n_cells) > 1:
                if np.all(n_cells == n_cells[0]) == False:
                    raise ValueError("Number of cells not match between modalities for batch " + str(batch))
            
            # No all None batch
            if np.all(np.array([self.Xs[mod][batch] is None for mod in self.mods]) == True) == True:
                raise ValueError("Don't have count matrix correspond to " + str(batch))
            
        # interaction matrix
        # extract all modes
        """ Not necessary to have interaction matrix for multi-modal data (vertical)
        all_mods = set(self.mods)
        interact_mods = set()
        for mods in self.As.keys():
            mod1, mod2 = mods.split("_")
            interact_mods.add(mod1)
            interact_mods.add(mod2)
        if (len(all_mods) > 1) and (interact_mods != all_mods):
            raise ValueError("Following modes are not connected through interaction matrices: " + str(all_mods))
        """
        # check dimension match between As matrix and feature matrix
        for mods in self.As.keys():
            mod1, mod2 = mods.split("_")
            n_features_mod1, n_features_mod2 = self.As[mods].shape
            if (n_features_mod1 != n_features[mod1][0]) | (n_features_mod2 != n_features[mod2][0]):
                raise ValueError("Number of features not match for interaction matrix")
            
            
        print("Finished.")
        
    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim = 1)
    
    @staticmethod
    def recon_loss(X, C1, C2, Sigma, b1, b2):
        return (X - C1 @ Sigma @ C2.t() - b1 - b2).pow(2).mean()
    
    @staticmethod
    def cosine_loss(A, B):
        return -torch.trace(A.t() @ B)/torch.norm(A)/torch.norm(B)

    def sample_mini_batch(self):
        """\
        Sample mini batch
        """
        mask_cells = []
        mask_feats = []
        # sample mini_batch for each cell dimension
        for batch in range(len(self.Ns)):
            mask_cells.append(np.random.choice(self.C_cells[batch].shape[0], int(self.C_cells[batch].shape[0] * self.batch_size), replace=False))
        
        # sample mini_batch for each feature dimension
        for idx_mod, mod in enumerate(self.mods):
            mask_feats.append(np.random.choice(self.C_feats[idx_mod].shape[0], int(self.C_feats[idx_mod].shape[0] * self.batch_size), replace=False))
        
        return mask_cells, mask_feats

    def batch_loss(self, mode, alpha, batch_indices = None):
        """\
            Calculate overall loss term
        """
        # init
        loss1 = 0
        loss2 = 0
        loss3 = 0
        loss4 = 0
        loss5 = 0

        if mode != 'valid':
            mask_cells = batch_indices["cells"]
            mask_feats = batch_indices["feats"]

        # reconstruction loss    
        for batch in range(len(self.Ns)):
            for idx_mod, mod in enumerate(self.mods):
                if self.Xs[mod][batch] is not None:
                    if mode != "valid":
                        batch_X = self.Xs[mod][batch][np.ix_(mask_cells[batch], mask_feats[idx_mod])]
                        batch_C_cells = self.C_cells[batch][mask_cells[batch],:]
                        batch_C_feats = self.C_feats[idx_mod][mask_feats[idx_mod], :]
                        batch_A_asso = self.A_assos[idx_mod]
                        batch_b_cells = self.b_cells[mod][batch][mask_cells[batch],:]
                        batch_b_feats = self.b_feats[mod][batch][:, mask_feats[idx_mod]]
                        loss1 += self.recon_loss(batch_X, self.softmax(batch_C_cells), self.softmax(batch_C_feats), batch_A_asso, batch_b_cells, batch_b_feats)
                    else:
                        loss1 += self.recon_loss(self.Xs[mod][batch], self.softmax(self.C_cells[batch]), self.softmax(self.C_feats[idx_mod]), self.A_assos[idx_mod], self.b_cells[mod][batch], self.b_feats[mod][batch])
            
            # for missing clusters, calculate when there is missing clusters, mode = "valid" or "C_cells"
            if len(self.missing_dims[batch]) > 0: 
                if mode != "valid":
                    loss2 += (self.softmax(self.C_cells[batch][mask_cells[batch],:])[:,self.missing_dims[batch]]).mean()
                elif mode == "C_cells":
                    loss2 += (self.softmax(self.C_cells[batch])[:,self.missing_dims[batch]]).mean()

        # association loss, calculate when mode is "valid" or "A_assos"
        if (mode in ["A_assos", "valid"]):
            for idx_mod, mod in enumerate(self.mods):
                if len(self.mods) == 1:
                    loss3 += 0
                elif idx_mod != len(self.mods) - 1:
                    loss3 += self.cosine_loss(self.A_assos[idx_mod+1], self.A_assos[idx_mod])
                else:
                    loss3 += self.cosine_loss(self.A_assos[0], self.A_assos[idx_mod])
                
                loss5 += self.A_assos[idx_mod].abs().sum()        
        
        # modality connection loss, calculate when mode is "valid" or "C_feats"    
        for mods in self.As.keys():
            mod1, mod2 = mods.split("_")
            idx_mod1 = np.where(np.array(self.mods) == mod1)[0][0]
            idx_mod2 = np.where(np.array(self.mods) == mod2)[0][0]
            if mode != "valid":
                batch_A = self.As[mods][np.ix_(mask_feats[idx_mod1], mask_feats[idx_mod2])]
                batch_C_feats1 = self.C_feats[idx_mod1][mask_feats[idx_mod1],:]
                batch_C_feats2 = self.C_feats[idx_mod2][mask_feats[idx_mod2],:]
                loss4 += self.cosine_loss(self.softmax(batch_C_feats1), batch_A @ self.softmax(batch_C_feats2))
            elif mode == "C_feats":
                loss4 += self.cosine_loss(self.softmax(self.C_feats[idx_mod1]), batch_A @ self.softmax(self.C_feats[idx_mod2]))

        loss = alpha[0] * loss1 + alpha[1] * loss2 + alpha[2] * loss3 + alpha[3] * loss4 + alpha[4] * loss5 

        return loss, alpha[0] * loss1, alpha[1] * loss2, alpha[2] * loss3, alpha[3] * loss4, alpha[4] * loss5
    

    def train_func(self, T):
        best_loss = 1e12
        count = 0
        losses = []

        for t in range(T):
            mask_cells, mask_feats = self.sample_mini_batch()
            
            # update C_cells
            for i, mod in enumerate(self.mods):
                self.C_feats[i].requires_grad = False
                self.A_assos[i].requires_grad = False
            for batch in range(len(self.Ns)):
                self.C_cells[batch].requires_grad = True
            loss, *_ = self.batch_loss(mode = "C_cells", alpha = self.alpha, batch_indices = {"cells": mask_cells, "feats": mask_feats})
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # update C_feats
            for batch in range(len(self.Ns)):
                self.C_cells[batch].requires_grad = False
            for i, mod in enumerate(self.mods):
                self.C_feats[i].requires_grad = True
                # update only one C_feats a time
                if i > 0:
                    self.C_feats[i-1].requires_grad = False
                loss, *_ = self.batch_loss(mode = "C_feats", alpha = self.alpha, batch_indices = {"cells": mask_cells, "feats": mask_feats})
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # update A_assos:
            self.C_feats[-1].requires_grad = False
            for i, mod in enumerate(self.mods):
                self.A_assos[i].requires_grad = True 
                # update only one A_assos a time
                if i > 0:
                    self.A_assos[i-1].requires_grad = False  
                loss, *_ = self.batch_loss(mode = "A_assos", alpha = self.alpha, batch_indices = {"cells": mask_cells, "feats": mask_feats})                    
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            # update bias term:
            
            with torch.no_grad():
                for batch in range(len(self.Ns)):
                    for idx_mod, mod in enumerate(self.mods):
                        if self.Xs[mod][batch] is not None:
                            batch_X = self.Xs[mod][batch][np.ix_(mask_cells[batch], mask_feats[idx_mod])]
                            batch_C_cells = self.C_cells[batch][mask_cells[batch],:]
                            batch_C_feats = self.C_feats[idx_mod][mask_feats[idx_mod], :]
                            batch_A_asso = self.A_assos[idx_mod]
                            batch_b_feats = self.b_feats[mod][batch][:, mask_feats[idx_mod]]
                            self.b_cells[mod][batch][mask_cells[batch], :] = torch.mean(batch_X - self.softmax(batch_C_cells) @ batch_A_asso @ self.softmax(batch_C_feats).t() - batch_b_feats, dim = 1)[:,None]
                            batch_b_cells = self.b_cells[mod][batch][mask_cells[batch],:]
                            self.b_feats[mod][batch][:, mask_feats[idx_mod]] = torch.mean(batch_X - self.softmax(batch_C_cells) @ batch_A_asso @ self.softmax(batch_C_feats).t() - batch_b_cells, dim = 0)[None,:]

            
            # validation       
            if (t+1) % self.interval == 0:
                with torch.no_grad():
                    loss, loss1, loss2, loss3, loss4, loss5 = self.batch_loss(mode = "valid", alpha = self.alpha)
                
                print('Epoch {}, Validating Loss: {:.4f}'.format(t + 1, loss.item()))
                info = [
                    'loss 1: {:.5f}'.format(loss1.item()),
                    'loss 2: {:.5f}'.format(loss2.item()),
                    'loss 3: {:.5f}'.format(loss3.item()),
                    'loss 4: {:.5f}'.format(loss4.item()),
                    'loss 5: {:.5f}'.format(loss5.item())
                ]
                for i in info:
                    print("\t", i)
                
                losses.append(loss.item())
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(self.state_dict(), f'../check_points/real_{self.N_cell}.pt')
                    count = 0
                else:
                    count += 1
                    if count % 20 == 0:
                        self.optimizer.param_groups[0]['lr'] *= 0.5
                        print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < 1e-6:
                            break
                        else:
                            self.load_state_dict(torch.load(f'../check_points/real_{self.N_cell}.pt'))
                            count = 0                            
         
        # return losses                            

class cfrm(Module):
    """\
        Gene clusters more than cell clusters, force A_r and A_g to be sparse:
        
        alpha[0]: the weight of the scRNA-Seq tri-factorization
        alpha[1]: the weight of the scATAC-Seq tri-factorization
        alpha[2]: the weight of the association relationship between A_r and A_g
        alpha[3]: the weight of the gene activity matrix
        alpha[4]: the missing clusters
        alpha[5]: the orthogonality of the cell embedding
        alpha[6]: the orthogonality of the feature embedding
        alpha[7]: the sparsity of A_r and A_g
        
    """
    def __init__(self, counts, Ns, K, N_feat = None, batch_size=0.3, interval=10, lr=1e-3, alpha = [1000, 1000, 100, 100, 0.0, 0.0, 0.0, 0.1], seed = None):
        super().__init__()
        
        # init parameters, Ns is a list with length the number of batches
        self.Ns = Ns
        self.K = K
        self.N_cell = sum(self.Ns) - (len(self.Ns) - 1) * self.K
        if N_feat is None:
            self.N_feat = self.N_cell + 1
        else:
            self.N_feat = N_feat
        
        self.batch_size = batch_size
        self.interval = interval
        self.alpha = torch.FloatTensor(alpha).to(device)
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # sanity check
        assert len(counts["rna"]) == len(counts["atac"])
        assert len(self.Ns) == len(counts["rna"])
        for i in range(len(self.Ns)):
            if (counts["rna"][i] is not None) and (counts["atac"][i] is not None):
                assert counts["rna"][i].shape[0] == counts["atac"][i].shape[0]
        
        
        # preprocessing data
        self.Gs = []
        for counts_rna in counts["rna"]:
            if counts_rna is not None:
                counts_rna = utils.preprocess(counts_rna, modality = "RNA")
                counts_rna = counts_rna/np.max(counts_rna)
                self.Gs.append(torch.FloatTensor(counts_rna).to(device))
            else:
                # if corresponding batch doesn't have scRNA-Seq, give None
                self.Gs.append(None)
            
        self.Rs = []
        for counts_atac in counts["atac"]:
            if counts_atac is not None:
                counts_atac = utils.preprocess(counts_atac, modality = "ATAC")
                counts_atac = counts_atac/np.max(counts_atac)
                self.Rs.append(torch.FloatTensor(counts_atac).to(device))
            else:
                # if corresponding batch doesn't have scATAC-Seq, give None
                self.Rs.append(None)
        
        # only one gene activity matrix for multi-batches scRNA-Seq and scATAC-Seq datasets.
        if counts["gact"] is not None:
            gact = counts["gact"]
            assert gact.shape[0] < gact.shape[1]
            gact = utils.preprocess(gact, modality = "interaction")
            self.A = torch.FloatTensor(gact).to(device)
        
        else:
            # if (len(counts["atac"]) != 0) and (len(counts["rna"]) != 0):
            if all([elem != None for elem in counts["atac"]]) and all(elem != None for elem in counts["rna"]):
                raise ValueError("gene activity matrix must be provided")
        
        # create parameters
        self.Cs = ParameterList([])        
        self.b1s = []
        self.bgs = []

        self.b2s = []
        self.brs = []
        
        self.Cg = None
        self.Cr = None
        self.Ag = None
        self.Ar = None
        
        for idx in range(len(self.Ns)):
            if self.Gs[idx] is not None:
                self.Cs.append(Parameter(torch.rand(self.Gs[idx].shape[0], self.N_cell)))
                    
            elif self.Rs[idx] is not None:
                self.Cs.append(Parameter(torch.rand(self.Rs[idx].shape[0], self.N_cell)))

            else:
                raise ValueError("counts_rna and counts_atac cannot be both None")
                
            if self.Gs[idx] is not None:
                self.b1s.append(torch.zeros(self.Gs[idx].shape[0], 1).to(device))
                self.bgs.append(torch.zeros(1, self.Gs[idx].shape[1]).to(device))
                if self.Cg is None:
                    self.Cg = Parameter(torch.rand(self.Gs[idx].shape[1], self.N_feat))
                if self.Ag is None:
                    self.Ag = Parameter(torch.rand((self.N_cell, self.N_feat)))
            else:
                self.b1s.append(None)
                self.bgs.append(None)
                
            if self.Rs[idx] is not None:
                self.b2s.append(torch.zeros(self.Rs[idx].shape[0], 1).to(device))
                self.brs.append(torch.zeros(1, self.Rs[idx].shape[1]).to(device))
                if self.Cr is None:
                    self.Cr = Parameter(torch.rand(self.Rs[idx].shape[1], self.N_feat))
                if self.Ar is None:
                    self.Ar = Parameter(torch.rand((self.N_cell, self.N_feat))) 
            else:
                self.b2s.append(None)
                self.brs.append(None)
        
        
        # missing masks
        self.dims = []
        self.missing_dims = []
        self.mask_diags = []
        K_comps = [N - self.K for N in self.Ns]
        point = 0
        for K_comp in K_comps:
            self.dims.append([x for x in range(self.K)] + [x for x in range(self.K + point, self.K + point + K_comp)])
            self.missing_dims.append([x for x in range(self.K, self.K + point)] + [x for x in range(self.K + point + K_comp, self.N_cell)])
            self.mask_diags.append(torch.eye(len(self.dims[-1])).to(device))
            point += K_comp
        
        self.mask_diagg = torch.eye(self.N_feat).to(device)
        self.mask_diagr = torch.eye(self.N_feat).to(device)
        
        self.optimizer = opt.Adam(self.parameters(), lr=lr)
        
        self.orders = ['C_12']*1 
        if self.Cg is not None:
            self.orders += ['C_g']*1
        if self.Cr is not None:
            self.orders += ['C_r']*1
        if self.Ag is not None:
            self.orders += ['A_g']*1
        if self.Ar is not None:
            self.orders += ['A_r']*1
        self.orders += ['b']* 1 
    
    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim = 1)
    
    def batch_loss(self, mode, alpha, batch_indices = None):
        # init
        loss1 = 0
        loss2 = 0
        loss3 = 0
        loss4 = 0
        loss5 = 0
        loss6 = 0
        loss7 = 0
        loss8 = 0

        if mode != 'valid':
            masks = batch_indices[0]
            mask_g = batch_indices[1]
            mask_r = batch_indices[2]

        if mode == 'C_12':
            if self.Cg is not None:
                self.Cg.requires_grad = False
            if self.Cr is not None:
                self.Cr.requires_grad = False
            if self.Ag is not None:
                self.Ag.requires_grad = False
            if self.Ar is not None:
                self.Ar.requires_grad = False
            
            for i in range(len(self.Ns)):
                self.Cs[i].requires_grad = True
                if self.Gs[i] is not None:
                    loss1 += (self.Gs[i][np.ix_(masks[i], mask_g)] - self.softmax(self.Cs[i][masks[i],:]) @ self.Ag @ self.softmax(self.Cg[mask_g, :]).t() - self.bgs[i][:, mask_g] - self.b1s[i][masks[i],:]).pow(2).mean()
                
            
                if self.Rs[i] is not None:
                    loss2 += (self.Rs[i][np.ix_(masks[i], mask_r)] - self.softmax(self.Cs[i][masks[i],:]) @ self.Ar @ self.softmax(self.Cr[mask_r, :]).t() - self.brs[i][:, mask_r] - self.b2s[i][masks[i],:]).pow(2).mean() 
                    
            
                if len(self.missing_dims[i]) > 0: 
                    loss5 += (self.softmax(self.Cs[i][masks[i],:])[:,self.missing_dims[i]]).mean()

                if alpha[5] != 0:
                    Corr1 = self.softmax(self.Cs[i][masks[i],:])[:,self.dims[i]].t() @ self.softmax(self.Cs[i][masks[i], :])[:,self.dims[i]]
                    loss6 -= torch.norm(Corr1 * self.mask_diags[i]) / torch.norm(Corr1)
            
            
        elif mode == 'C_g':
            if self.Cg is not None:
                self.Cg.requires_grad = True
            if self.Cr is not None:
                self.Cr.requires_grad = False
            if self.Ag is not None:
                self.Ag.requires_grad = False
            if self.Ar is not None:
                self.Ar.requires_grad = False
            
            for i in range(len(self.Ns)):
                self.Cs[i].requires_grad = False
                if self.Gs[i] is not None:
                    loss1 += (self.Gs[i][np.ix_(masks[i], mask_g)] - self.softmax(self.Cs[i][masks[i],:]) @ self.Ag @ self.softmax(self.Cg[mask_g, :]).t() - self.bgs[i][:, mask_g] - self.b1s[i][masks[i],:]).pow(2).mean()
                
            
            if (self.Cg is not None) and (self.Cr is not None):
                loss4 = - torch.trace(self.softmax(self.Cg[mask_g,:]).t() @ self.A[np.ix_(mask_g, mask_r)] @ self.softmax(self.Cr[mask_r,:])) /torch.norm(self.softmax(self.Cg[mask_g,:])) / torch.norm(self.A[np.ix_(mask_g, mask_r)] @ self.softmax(self.Cr[mask_r,:]))
                    
            if (alpha[6] != 0) and (self.Cg is not None):
                Corrg = self.softmax(self.Cg[mask_g,:]).t() @ self.softmax(self.Cg[mask_g,:])
                loss7 -= torch.norm(Corrg * self.mask_diagg) / torch.norm(Corrg)                 
                
                
        elif mode == 'C_r':
            if self.Cg is not None:
                self.Cg.requires_grad = False
            if self.Cr is not None:
                self.Cr.requires_grad = True
            if self.Ag is not None:
                self.Ag.requires_grad = False
            if self.Ar is not None:
                self.Ar.requires_grad = False
            
            for i in range(len(self.Ns)):
                self.Cs[i].requires_grad = False
                if self.Rs[i] is not None:
                    loss2 += (self.Rs[i][np.ix_(masks[i], mask_r)] - self.softmax(self.Cs[i][masks[i],:]) @ self.Ar @ self.softmax(self.Cr[mask_r, :]).t() - self.brs[i][:, mask_r] - self.b2s[i][masks[i],:]).pow(2).mean()
            
            
            if (self.Cg is not None) and (self.Cr is not None):
                loss4 = - torch.trace(self.softmax(self.Cg[mask_g,:]).t() @ self.A[np.ix_(mask_g, mask_r)] @ self.softmax(self.Cr[mask_r,:])) /torch.norm(self.softmax(self.Cg[mask_g,:])) / torch.norm(self.A[np.ix_(mask_g, mask_r)] @ self.softmax(self.Cr[mask_r,:]))               
                
            if (alpha[6] != 0) and (self.Cr is not None):    
                Corrr = self.softmax(self.Cr[mask_r,:]).t() @ self.softmax(self.Cr[mask_r,:])
                loss7 -= torch.norm(Corrr * self.mask_diagr) / torch.norm(Corrr)
            
            
        elif mode == "A_g":
            if self.Cg is not None:
                self.Cg.requires_grad = False
            if self.Cr is not None:
                self.Cr.requires_grad = False
            if self.Ag is not None:
                self.Ag.requires_grad = True
            if self.Ar is not None:
                self.Ar.requires_grad = False
            
            for i in range(len(self.Ns)):
                self.Cs[i].requires_grad = False 
                if self.Gs[i] is not None:
                    loss1 += (self.Gs[i][np.ix_(masks[i], mask_g)] - self.softmax(self.Cs[i][masks[i],:]) @ self.Ag @ self.softmax(self.Cg[mask_g,:]).t() - self.bgs[i][:, mask_g] - self.b1s[i][masks[i],:]).pow(2).mean()
                    
                
            if (self.Ag is not None) and (self.Ar is not None):
                loss3 = - torch.trace(self.Ar @ self.Ag.t())/torch.norm(self.Ar)/torch.norm(self.Ag)
            
            if self.Ag is not None:
                loss8 = self.Ag.abs().sum()

        
        elif mode == "A_r":
            if self.Cg is not None:
                self.Cg.requires_grad = False
            if self.Cr is not None:
                self.Cr.requires_grad = False
            if self.Ag is not None:
                self.Ag.requires_grad = False
            if self.Ar is not None:
                self.Ar.requires_grad = True
                
            for i in range(len(self.Ns)):
                self.Cs[i].requires_grad = False 
                if self.Rs[i] is not None:
                    loss2 += (self.Rs[i][np.ix_(masks[i], mask_r)] - self.softmax(self.Cs[i][masks[i],:]) @ self.Ar @ self.softmax(self.Cr[mask_r, :]).t() - self.brs[i][:, mask_r] - self.b2s[i][masks[i],:]).pow(2).mean()
                    
            
            if (self.Ag is not None) and (self.Ar is not None):                
                loss3 = - torch.trace(self.Ar @ self.Ag.t())/torch.norm(self.Ar)/torch.norm(self.Ag)
            
            if self.Ar is not None:
                loss8 = self.Ar.abs().sum()
            
            
        elif mode == "b":
            with torch.no_grad():
                for i in range(len(self.Ns)):
                    if self.Gs[i] is not None:
                        self.bgs[i][:, mask_g] = torch.mean(self.Gs[i][np.ix_(masks[i], mask_g)] - self.softmax(self.Cs[i][masks[i],:]) @ self.Ag @ self.softmax(self.Cg[mask_g, :]).t() - self.b1s[i][masks[i], :], dim = 0)[None,:]
                        self.b1s[i][masks[i], :] = torch.mean(self.Gs[i][np.ix_(masks[i], mask_g)] - self.softmax(self.Cs[i][masks[i],:]) @ self.Ag @ self.softmax(self.Cg[mask_g, :]).t() - self.bgs[i][:, mask_g], dim = 1)[:,None]
                
                    if self.Rs[i] is not None:
                        self.brs[i][:, mask_r] = torch.mean(self.Rs[i][np.ix_(masks[i], mask_r)] - self.softmax(self.Cs[i][masks[i],:]) @ self.Ar @ self.softmax(self.Cr[mask_r, :]).t() - self.b2s[i][masks[i],:], dim = 0)[None,:]
                        self.b2s[i][masks[i], :] = torch.mean(self.Rs[i][np.ix_(masks[i], mask_r)] - self.softmax(self.Cs[i][masks[i],:]) @ self.Ar @ self.softmax(self.Cr[mask_r, :]).t() - self.brs[i][:, mask_r], dim = 1)[:,None]
            
                
        elif mode == 'valid':
            with torch.no_grad():
                
                for i in range(len(self.Ns)):
                    if self.Gs[i] is not None:
                        loss1 += (self.Gs[i] - self.softmax(self.Cs[i]) @ self.Ag @ self.softmax(self.Cg).t() - self.bgs[i] - self.b1s[i]).pow(2).mean()
                    
                    if self.Rs[i] is not None:
                        loss2 += (self.Rs[i] - self.softmax(self.Cs[i]) @ self.Ar @ self.softmax(self.Cr).t() - self.brs[i] - self.b2s[i]).pow(2).mean()
                    
                    if len(self.missing_dims[i]) > 0: 
                        loss5 += (self.softmax(self.Cs[i])[:,self.missing_dims[i]]).mean()
                        
                    if alpha[5] != 0:
                        Corr1 = self.softmax(self.Cs[i])[:,self.dims[i]].t() @ self.softmax(self.Cs[i])[:,self.dims[i]]
                        loss6 -= torch.norm(Corr1 * self.mask_diags[i]) / torch.norm(Corr1)
                
                
                if (self.Cg is not None) and (self.Cr is not None):
                    loss3 = - torch.trace(self.Ar @ self.Ag.t())/torch.norm(self.Ar)/torch.norm(self.Ag)
                    loss4 = - torch.trace(self.softmax(self.Cg).t() @ self.A @ self.softmax(self.Cr)) /torch.norm(self.softmax(self.Cg)) / torch.norm(self.A @ self.softmax(self.Cr))

                
                if self.Cg is not None:
                    Corrg = self.softmax(self.Cg).t() @ self.softmax(self.Cg)
                    loss7 -= torch.norm(Corrg * self.mask_diagg) / torch.norm(Corrg)
                    loss8 += self.Ag.abs().sum()
                    
                if self.Cr is not None:
                    Corrr = self.softmax(self.Cr).t() @ self.softmax(self.Cr)
                    loss7 -= torch.norm(Corrr * self.mask_diagr) / torch.norm(Corrr)
                    loss8 += self.Ar.abs().sum()
                
        else:
            raise NotImplementedError
        
        loss = alpha[0] * loss1 + alpha[1] * loss2 + alpha[2] * loss3 + alpha[3] * loss4 + alpha[4] * loss5 + alpha[5] * loss6 + alpha[6] * loss7 + alpha[7] * loss8

        return loss, alpha[0] * loss1, alpha[1] * loss2, alpha[2] * loss3, alpha[3] * loss4, alpha[4] * loss5, alpha[5] * loss6, alpha[6] * loss7, alpha[7] * loss8
    
    def train_func(self, T):
        best_loss = 1e12
        count = 0

        T1 = int(T/4)
            
        alpha = self.alpha.clone()
        alpha[5] = 0
        alpha[6] = 0

        for t in range(T):
            # generate random masks
            masks = []
            
            for i in range(len(self.Ns)):
                masks.append(np.random.choice(self.Cs[i].shape[0], int(self.Cs[i].shape[0] * self.batch_size), replace=False))
            
            if self.Cg is not None:
                mask_g = np.random.choice(self.Cg.shape[0], int(self.Cg.shape[0] * self.batch_size), replace=False)
            else:
                mask_g = None
                
            if self.Cr is not None:
                mask_r = np.random.choice(self.Cr.shape[0], int(self.Cr.shape[0] * self.batch_size), replace=False)
            else:
                mask_r = None
            
            # update alpha with self.alpha
            if t == T1:
                alpha = self.alpha
            
            for mode in self.orders:
                loss, *_ = self.batch_loss(mode, alpha, [masks, mask_g, mask_r])
                
                if mode != 'b':
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                
            if (t+1) % self.interval == 0:
                
                loss, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8 = self.batch_loss('valid', alpha)
                
                print('Epoch {}, Validating Loss: {:.4f}'.format(t + 1, loss.item()))
                info = [
                    'loss 1: {:.5f}'.format(loss1.item()),
                    'loss 2: {:.5f}'.format(loss2.item()),
                    'loss 3: {:.5f}'.format(loss3.item()),
                    'loss 4: {:.5f}'.format(loss4.item()),
                    'loss 5: {:.5f}'.format(loss5.item()),
                    'loss 6: {:.5f}'.format(loss6.item()),
                    'loss 7: {:.5f}'.format(loss7.item()),
                    'loss 8: {:.5f}'.format(loss8.item())
                ]
                for i in info:
                    print("\t", i)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(self.state_dict(), f'../check_points/real_{self.N_cell}.pt')
                    count = 0
                else:
                    count += 1
                    if count % 20 == 0:
                        self.optimizer.param_groups[0]['lr'] *= 0.5
                        print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < 1e-6:
                            break
                        else:
                            self.load_state_dict(torch.load(f'../check_points/real_{self.N_cell}.pt'))
                            count = 0


# class cfrm_new(Module):
#     """\
#         Gene clusters more than cell clusters, force A_r and A_g to be sparse:
        
#         alpha[0]: the weight of the scRNA-Seq tri-factorization
#         alpha[1]: the weight of the scATAC-Seq tri-factorization
#         alpha[2]: the weight of the association relationship between A_r and A_g
#         alpha[3]: the weight of the gene activity matrix
#         alpha[4]: the missing clusters
#         alpha[5]: the orthogonality of the cell embedding
#         alpha[6]: the orthogonality of the feature embedding
#         alpha[7]: the sparsity of A_r and A_g
        
#     """
#     def __init__(self, counts, Ns, K, N_feat = None, batch_size=0.3, interval=10, lr=1e-3, alpha = [1000, 1000, 100, 100, 0.0, 0.0, 0.0, 0.1], seed = None):
#         super().__init__()
        
#         # init parameters, Ns is a list with length the number of batches
#         self.Ns = Ns
#         self.K = K
#         self.N_cell = sum(self.Ns) - (len(self.Ns) - 1) * self.K
#         if N_feat is None:
#             self.N_feat = self.N_cell + 1
#         else:
#             self.N_feat = N_feat
        
#         self.batch_size = batch_size
#         self.interval = interval
#         self.alpha = torch.FloatTensor(alpha).to(device)
        
#         if seed is not None:
#             np.random.seed(seed)
#             torch.manual_seed(seed)
        
#         # sanity check
#         assert len(counts["rna"]) == len(counts["atac"])
#         assert len(self.Ns) == len(counts["rna"])
#         for i in range(len(self.Ns)):
#             if (counts["rna"][i] is not None) and (counts["atac"][i] is not None):
#                 assert counts["rna"][i].shape[0] == counts["atac"][i].shape[0]
        
        
#         # preprocessing data
#         self.Gs = []
#         for counts_rna in counts["rna"]:
#             if counts_rna is not None:
#                 counts_rna = utils.preprocess(counts_rna, mode = "quantile", modality = "RNA")
#                 counts_rna = counts_rna/np.max(counts_rna)
#                 self.Gs.append(torch.FloatTensor(counts_rna).to(device))
#             else:
#                 # if corresponding batch doesn't have scRNA-Seq, give None
#                 self.Gs.append(None)
            
#         self.Rs = []
#         for counts_atac in counts["atac"]:
#             if counts_atac is not None:
#                 counts_atac = utils.preprocess(counts_atac, mode = "quantile", modality = "ATAC")
#                 counts_atac = counts_atac/np.max(counts_atac)
#                 self.Rs.append(torch.FloatTensor(counts_atac).to(device))
#             else:
#                 # if corresponding batch doesn't have scATAC-Seq, give None
#                 self.Rs.append(None)
        
#         # only one gene activity matrix for multi-batches scRNA-Seq and scATAC-Seq datasets.
#         if counts["gact"] is not None:
#             gact = counts["gact"]
#             assert gact.shape[0] < gact.shape[1]
#             gact = utils.preprocess(gact, mode = "gact")
#             self.A = torch.FloatTensor(gact).to(device)
        
#         else:
#             # if (len(counts["atac"]) != 0) and (len(counts["rna"]) != 0):
#             if all([elem != None for elem in counts["atac"]]) and all(elem != None for elem in counts["rna"]):
#                 raise ValueError("gene activity matrix must be provided")
        
#         # create parameters
#         self.Cs = ParameterList([])   
#         self.Vg = ParameterList([])
#         self.Vr = ParameterList([])
        
#         self.Cg = None
#         self.Cr = None
#         self.Ag = None
#         self.Ar = None
        
#         for idx in range(len(self.Ns)):
#             if self.Gs[idx] is not None:
#                 self.Cs.append(Parameter(torch.rand(self.Gs[idx].shape[0], self.N_cell)))
                    
#             elif self.Rs[idx] is not None:
#                 self.Cs.append(Parameter(torch.rand(self.Rs[idx].shape[0], self.N_cell)))

#             else:
#                 raise ValueError("counts_rna and counts_atac cannot be both None")
                
#             if self.Gs[idx] is not None:
#                 self.Vg.append(Parameter(torch.rand(self.Gs[idx].shape[1], self.N_cell)))
#                 if self.Cg is None:
#                     self.Cg = Parameter(torch.rand(self.Gs[idx].shape[1], self.N_feat))
#                 if self.Ag is None:
#                     self.Ag = Parameter(torch.rand((self.N_cell, self.N_feat)))
#             else:
#                 self.Vg.append(None)
                
#             if self.Rs[idx] is not None:
#                 self.Vr.append(Parameter(torch.rand(self.Rs[idx].shape[1], self.N_cell)))
#                 if self.Cr is None:
#                     self.Cr = Parameter(torch.rand(self.Rs[idx].shape[1], self.N_feat))
#                 if self.Ar is None:
#                     self.Ar = Parameter(torch.rand((self.N_cell, self.N_feat))) 
#             else:
#                 self.Vr.append(None)

        
        
#         # missing masks
#         self.dims = []
#         self.missing_dims = []
#         self.mask_diags = []
#         K_comps = [N - self.K for N in self.Ns]
#         point = 0
#         for K_comp in K_comps:
#             self.dims.append([x for x in range(self.K)] + [x for x in range(self.K + point, self.K + point + K_comp)])
#             self.missing_dims.append([x for x in range(self.K, self.K + point)] + [x for x in range(self.K + point + K_comp, self.N_cell)])
#             self.mask_diags.append(torch.eye(len(self.dims[-1])).to(device))
#             point += K_comp
        
#         self.mask_diagg = torch.eye(self.N_feat).to(device)
#         self.mask_diagr = torch.eye(self.N_feat).to(device)
        
#         self.optimizer = opt.Adam(self.parameters(), lr=lr)
        
#         self.orders = ['C_12']*1 
#         if self.Cg is not None:
#             self.orders += ['C_g']*1
#         if self.Cr is not None:
#             self.orders += ['C_r']*1
#         if self.Ag is not None:
#             self.orders += ['A_g']*1
#         if self.Ar is not None:
#             self.orders += ['A_r']*1
    
#     @staticmethod
#     def softmax(X: Tensor):
#         return torch.softmax(X, dim = 1)
    
#     def batch_loss(self, mode, alpha, batch_indices = None):
#         # init
#         loss1 = 0
#         loss2 = 0
#         loss3 = 0
#         loss4 = 0
#         loss5 = 0
#         loss6 = 0
#         loss7 = 0
#         loss8 = 0
#         loss9 = 0

#         if mode != 'valid':
#             masks = batch_indices[0]
#             mask_g = batch_indices[1]
#             mask_r = batch_indices[2]

#         if mode == 'C_12':
#             if self.Cg is not None:
#                 self.Cg.requires_grad = False
#             if self.Cr is not None:
#                 self.Cr.requires_grad = False
#                 self.Vg.requires_grad = False
#             if self.Ag is not None:
#                 self.Ag.requires_grad = False
#             if self.Ar is not None:
#                 self.Ar.requires_grad = False
            
            
            
#             for i in range(len(self.Ns)):
#                 self.Cs[i].requires_grad = True
#                 if self.Vg[i] is not None:
#                     self.Vg[i].requires_grad = False
#                 if self.Vr[i] is not None:
#                     self.Vr[i].requires_grad = False
                
#                 if self.Gs[i] is not None:
#                     loss1 += (self.Gs[i][np.ix_(masks[i], mask_g)] - self.softmax(self.Cs[i][masks[i],:]) @ self.Ag @ self.softmax(self.Cg[mask_g, :]).t() - self.softmax(self.Cs[i][masks[i],:]) @ self.Vg[i][mask_g, :].t()).pow(2).mean()
#                     loss6 += (self.softmax(self.Cs[i][masks[i],:]) @ self.Vg[i][mask_g, :].t()).pow(2).mean()
                
            
#                 if self.Rs[i] is not None:
#                     loss2 += (self.Rs[i][np.ix_(masks[i], mask_r)] - self.softmax(self.Cs[i][masks[i],:]) @ self.Ar @ self.softmax(self.Cr[mask_r, :]).t() - self.softmax(self.Cs[i][masks[i],:]) @ self.Vr[i][mask_r, :].t()).pow(2).mean()
#                     loss7 += (self.softmax(self.Cs[i][masks[i],:]) @ self.Vr[i][mask_r, :].t()).pow(2).mean()
                    
            
#                 if len(self.missing_dims[i]) > 0: 
#                     loss5 += (self.softmax(self.Cs[i][masks[i],:])[:,self.missing_dims[i]]).mean()

            
#         elif mode == 'C_g':
#             if self.Cg is not None:
#                 self.Cg.requires_grad = True
#             if self.Cr is not None:
#                 self.Cr.requires_grad = False
#             if self.Ag is not None:
#                 self.Ag.requires_grad = False
#             if self.Ar is not None:
#                 self.Ar.requires_grad = False
            
#             for i in range(len(self.Ns)):
#                 self.Cs[i].requires_grad = False
#                 if self.Vg[i] is not None:
#                     self.Vg[i].requires_grad = True
#                 if self.Vr[i] is not None:
#                     self.Vr[i].requires_grad = True
                
#                 if self.Gs[i] is not None:
#                     loss1 += (self.Gs[i][np.ix_(masks[i], mask_g)] - self.softmax(self.Cs[i][masks[i],:]) @ self.Ag @ self.softmax(self.Cg[mask_g, :]).t() - self.softmax(self.Cs[i][masks[i],:]) @ self.Vg[i][mask_g, :].t()).pow(2).mean()
#                     loss6 += (self.softmax(self.Cs[i][masks[i],:]) @ self.Vg[i][mask_g, :].t()).pow(2).mean()
                
            
#             if (self.Cg is not None) and (self.Cr is not None):
#                 loss4 = - torch.trace(self.softmax(self.Cg[mask_g,:]).t() @ self.A[np.ix_(mask_g, mask_r)] @ self.softmax(self.Cr[mask_r,:])) /torch.norm(self.softmax(self.Cg[mask_g,:])) / torch.norm(self.A[np.ix_(mask_g, mask_r)] @ self.softmax(self.Cr[mask_r,:]))
            
                
#         elif mode == 'C_r':
#             if self.Cg is not None:
#                 self.Cg.requires_grad = False
#             if self.Cr is not None:
#                 self.Cr.requires_grad = True
#             if self.Ag is not None:
#                 self.Ag.requires_grad = False
#             if self.Ar is not None:
#                 self.Ar.requires_grad = False
            
#             for i in range(len(self.Ns)):
#                 self.Cs[i].requires_grad = False
#                 if self.Rs[i] is not None:
#                     loss2 += (self.Rs[i][np.ix_(masks[i], mask_r)] - self.softmax(self.Cs[i][masks[i],:]) @ self.Ar @ self.softmax(self.Cr[mask_r, :]).t() - self.softmax(self.Cs[i][masks[i],:]) @ self.Vr[i][mask_r, :].t()).pow(2).mean()
#                     loss7 += (self.softmax(self.Cs[i][masks[i],:]) @ self.Vr[i][mask_r, :].t()).pow(2).mean()
            
            
#             if (self.Cg is not None) and (self.Cr is not None):
#                 loss4 = - torch.trace(self.softmax(self.Cg[mask_g,:]).t() @ self.A[np.ix_(mask_g, mask_r)] @ self.softmax(self.Cr[mask_r,:])) /torch.norm(self.softmax(self.Cg[mask_g,:])) / torch.norm(self.A[np.ix_(mask_g, mask_r)] @ self.softmax(self.Cr[mask_r,:]))               
            
            
#         elif mode == "A_g":
#             if self.Cg is not None:
#                 self.Cg.requires_grad = False
#             if self.Cr is not None:
#                 self.Cr.requires_grad = False
#             if self.Ag is not None:
#                 self.Ag.requires_grad = True
#             if self.Ar is not None:
#                 self.Ar.requires_grad = False
            
#             for i in range(len(self.Ns)):
#                 self.Cs[i].requires_grad = False 
#                 if self.Gs[i] is not None:
#                     loss1 += (self.Gs[i][np.ix_(masks[i], mask_g)] - self.softmax(self.Cs[i][masks[i],:]) @ self.Ag @ self.softmax(self.Cg[mask_g,:]).t() - self.softmax(self.Cs[i][masks[i],:]) @ self.Vg[i][mask_g, :].t()).pow(2).mean()
#                     loss6 += (self.softmax(self.Cs[i][masks[i],:]) @ self.Vg[i][mask_g, :].t()).pow(2).mean()
                    
                
#             if (self.Ag is not None) and (self.Ar is not None):
#                 loss3 = - torch.trace(self.Ar @ self.Ag.t())/torch.norm(self.Ar)/torch.norm(self.Ag)
            
#             if self.Ag is not None:
#                 loss8 = self.Ag.abs().sum()

        
#         elif mode == "A_r":
#             if self.Cg is not None:
#                 self.Cg.requires_grad = False
#             if self.Cr is not None:
#                 self.Cr.requires_grad = False
#             if self.Ag is not None:
#                 self.Ag.requires_grad = False
#             if self.Ar is not None:
#                 self.Ar.requires_grad = True
                
#             for i in range(len(self.Ns)):
#                 self.Cs[i].requires_grad = False 
#                 if self.Rs[i] is not None:
#                     loss2 += (self.Rs[i][np.ix_(masks[i], mask_r)] - self.softmax(self.Cs[i][masks[i],:]) @ self.Ar @ self.softmax(self.Cr[mask_r, :]).t() - self.softmax(self.Cs[i][masks[i],:]) @ self.Vr[i][mask_r, :].t()).pow(2).mean()
#                     loss7 += (self.softmax(self.Cs[i][masks[i],:]) @ self.Vr[i][mask_r, :].t()).pow(2).mean()
                    
            
#             if (self.Ag is not None) and (self.Ar is not None):                
#                 loss3 = - torch.trace(self.Ar @ self.Ag.t())/torch.norm(self.Ar)/torch.norm(self.Ag)
            
#             if self.Ar is not None:
#                 loss8 = self.Ar.abs().sum()
               
                
#         elif mode == 'valid':
#             with torch.no_grad():
                
#                 for i in range(len(self.Ns)):
#                     if self.Gs[i] is not None:
#                         loss1 += (self.Gs[i] - self.softmax(self.Cs[i]) @ self.Ag @ self.softmax(self.Cg).t() - self.softmax(self.Cs[i]) @ self.Vg[i].t()).pow(2).mean()
#                         loss6 += (self.softmax(self.Cs[i]) @ self.Vg[i].t()).pow(2).mean()
                    
#                     if self.Rs[i] is not None:
#                         loss2 += (self.Rs[i] - self.softmax(self.Cs[i]) @ self.Ar @ self.softmax(self.Cr).t() - self.softmax(self.Cs[i]) @ self.Vr[i].t()).pow(2).mean()
#                         loss7 += (self.softmax(self.Cs[i]) @ self.Vr[i].t()).pow(2).mean()
                    
#                     if len(self.missing_dims[i]) > 0: 
#                         loss5 += (self.softmax(self.Cs[i])[:,self.missing_dims[i]]).mean()
                
                
#                 if (self.Cg is not None) and (self.Cr is not None):
#                     loss3 = - torch.trace(self.Ar @ self.Ag.t())/torch.norm(self.Ar)/torch.norm(self.Ag)
#                     loss4 = - torch.trace(self.softmax(self.Cg).t() @ self.A @ self.softmax(self.Cr)) /torch.norm(self.softmax(self.Cg)) / torch.norm(self.A @ self.softmax(self.Cr))

                
#                 if self.Cg is not None:
#                     loss8 += self.Ag.abs().sum()
                    
#                 if self.Cr is not None:
#                     loss8 += self.Ar.abs().sum()
                
#         else:
#             raise NotImplementedError
        
#         loss = alpha[0] * loss1 + alpha[1] * loss2 + alpha[2] * loss3 + alpha[3] * loss4 + alpha[4] * loss5 + alpha[5] * loss6 + alpha[6] * loss7 + alpha[7] * loss8

#         return loss, alpha[0] * loss1, alpha[1] * loss2, alpha[2] * loss3, alpha[3] * loss4, alpha[4] * loss5, alpha[5] * loss6, alpha[6] * loss7, alpha[7] * loss8
    
#     def train_func(self, T):
#         best_loss = 1e12
#         count = 0

#         for t in range(T):
#             # generate random masks
#             masks = []
            
#             for i in range(len(self.Ns)):
#                 masks.append(np.random.choice(self.Cs[i].shape[0], int(self.Cs[i].shape[0] * self.batch_size), replace=False))
            
#             if self.Cg is not None:
#                 mask_g = np.random.choice(self.Cg.shape[0], int(self.Cg.shape[0] * self.batch_size), replace=False)
#             else:
#                 mask_g = None
                
#             if self.Cr is not None:
#                 mask_r = np.random.choice(self.Cr.shape[0], int(self.Cr.shape[0] * self.batch_size), replace=False)
#             else:
#                 mask_r = None
            
            
#             for mode in self.orders:
#                 loss, *_ = self.batch_loss(mode, self.alpha, [masks, mask_g, mask_r])
                
#                 if mode != 'b':
#                     loss.backward()
#                     self.optimizer.step()
#                     self.optimizer.zero_grad()
                
                
#             if (t+1) % self.interval == 0:
                
#                 loss, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8 = self.batch_loss('valid', self.alpha)
                
#                 print('Epoch {}, Validating Loss: {:.4f}'.format(t + 1, loss.item()))
#                 info = [
#                     'loss 1: {:.5f}'.format(loss1.item()),
#                     'loss 2: {:.5f}'.format(loss2.item()),
#                     'loss 3: {:.5f}'.format(loss3.item()),
#                     'loss 4: {:.5f}'.format(loss4.item()),
#                     'loss 5: {:.5f}'.format(loss5.item()),
#                     'loss 6: {:.5f}'.format(loss6.item()),
#                     'loss 7: {:.5f}'.format(loss7.item()),
#                     'loss 8: {:.5f}'.format(loss8.item())
#                 ]
#                 for i in info:
#                     print("\t", i)

#                 if loss.item() < best_loss:
#                     best_loss = loss.item()
#                     torch.save(self.state_dict(), f'../check_points/real_{self.N_cell}.pt')
#                     count = 0
#                 else:
#                     count += 1
#                     if count % 20 == 0:
#                         self.optimizer.param_groups[0]['lr'] *= 0.5
#                         print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
#                         if self.optimizer.param_groups[0]['lr'] < 1e-6:
#                             break
#                         else:
#                             self.load_state_dict(torch.load(f'../check_points/real_{self.N_cell}.pt'))
#                             count = 0

