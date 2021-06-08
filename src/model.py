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
        
        # init parameters, first for counts["rna"], then for counts["atac"]
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

        # preprocessing data
        self.Gs = []
        for counts_rna in counts["rna"]:
            counts_rna = utils.preprocess(counts_rna, mode = "quantile", modality = "RNA")
            counts_rna = counts_rna/np.max(counts_rna)
            self.Gs.append(torch.FloatTensor(counts_rna).to(device))
            
        self.Rs = []
        for counts_atac in counts["atac"]:
            counts_atac = utils.preprocess(counts_atac, mode = "quantile", modality = "ATAC")
            counts_atac = counts_atac/np.max(counts_atac)
            self.Rs.append(torch.FloatTensor(counts_atac).to(device))
        
        # only one gene activity matrix for multi-batches scRNA-Seq and scATAC-Seq datasets.
        gact = counts["gact"][0]
        assert gact.shape[0] < gact.shape[1]
        gact = utils.preprocess(gact, mode = "gact")
        self.A = torch.FloatTensor(gact).to(device)
        
        # create parameters
        self.C1s = ParameterList([])
        self.b1s = []
        self.bgs = []

        self.C2s = ParameterList([])
        self.b2s = []
        self.brs = []
        
        if len(self.Gs) != 0:
            for G in self.Gs:
                self.C1s.append(Parameter(torch.rand(G.shape[0], self.N_cell)))
                # not shared?
                self.b1s.append(torch.zeros(G.shape[0],1).to(device))
                self.bgs.append(torch.zeros(1, G.shape[1]).to(device))
                
            self.Cg = Parameter(torch.rand(self.Gs[0].shape[1], self.N_feat))
            self.Ag = Parameter(torch.rand((self.N_cell, self.N_feat)))
            
        else:
            self.Cg = None
            self.Ag = None
        
        if len(self.Rs) != 0:
            for R in self.Rs:
                self.C2s.append(Parameter(torch.rand(R.shape[0], self.N_cell)))
                # not shared?
                self.b2s.append(torch.zeros(R.shape[0], 1).to(device))
                self.brs.append(torch.zeros(1, R.shape[1]).to(device))

            self.Cr = Parameter(torch.rand(self.Rs[0].shape[1], self.N_feat))
            self.Ar = Parameter(torch.rand((self.N_cell, self.N_feat))) 
        else:
            self.Cr = None
            self.Ar = None
        
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
        
        self.orders = ['C_12']*1 + ['C_gr']*1
        if (len(self.Gs) != 0): 
            self.orders += ['A_g']*1
        if (len(self.Rs) != 0):
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
            mask_1s = batch_indices[0]
            mask_g = batch_indices[1]
            mask_2s = batch_indices[2]
            mask_r = batch_indices[3]

        if mode == 'C_12':

            # if no G, skip
            for i, G in enumerate(self.Gs):
                self.C1s[i].requires_grad = True
                self.Cg.requires_grad = False
                self.Ag.requires_grad = False
                
                loss1 += (G[np.ix_(mask_1s[i], mask_g)] - self.softmax(self.C1s[i][mask_1s[i],:]) @ self.Ag @ self.softmax(self.Cg[mask_g, :]).t() - self.bgs[i][:, mask_g] - self.b1s[i][mask_1s[i],:]).pow(2).mean()
                
                if len(self.missing_dims[i]) > 0: 
                    loss5 += (self.softmax(self.C1s[i][mask_1s[i],:])[:,self.missing_dims[i]]).mean()
                
                if alpha[5] != 0:
                    Corr1 = self.softmax(self.C1s[i][mask_1s[i],:])[:,self.dims[i]].t() @ self.softmax(self.C1s[i][mask_1s[i], :])[:,self.dims[i]]
                    loss6 -= torch.norm(Corr1 * self.mask_diags[i]) / torch.norm(Corr1)
                 
            # if no R, skip
            for i, R in enumerate(self.Rs):
                self.C2s[i].requires_grad = True
                self.Cr.requires_grad = False
                self.Ar.requires_grad = False
                
                loss2 += (R[np.ix_(mask_2s[i], mask_r)] - self.softmax(self.C2s[i][mask_2s[i],:]) @ self.Ar @ self.softmax(self.Cr[mask_r, :]).t() - self.brs[i][:, mask_r] - self.b2s[i][mask_2s[i],:]).pow(2).mean() 
                
                if len(self.missing_dims[i + len(self.Gs)]) > 0: 
                    loss5 += (self.softmax(self.C2s[i][mask_2s[i],:])[:,self.missing_dims[i + len(self.Gs)]]).mean()
                
                if alpha[5] != 0:
                    Corr2 = self.softmax(self.C2s[i][mask_2s[i],:])[:,self.dims[i + len(self.Gs)]].t() @ self.softmax(self.C2s[i][mask_2s[i], :])[:,self.dims[i + len(self.Gs)]]
                    loss6 -= torch.trace(Corr2 * self.mask_diags[i + len(self.Gs)]) / torch.norm(Corr2)

        
            
        elif mode == 'C_gr':
            
            for i, G in enumerate(self.Gs):
                self.C1s[i].requires_grad = False
                self.Cg.requires_grad = True
                self.Ag.requires_grad = False
                
                loss1 += (G[np.ix_(mask_1s[i], mask_g)] - self.softmax(self.C1s[i][mask_1s[i],:]) @ self.Ag @ self.softmax(self.Cg[mask_g, :]).t() - self.bgs[i][:, mask_g] - self.b1s[i][mask_1s[i],:]).pow(2).mean()
                
            for i, R in enumerate(self.Rs):
                self.C2s[i].requires_grad = False
                self.Cr.requires_grad = True
                self.Ar.requires_grad = False
                
                loss2 += (R[np.ix_(mask_2s[i], mask_r)] - self.softmax(self.C2s[i][mask_2s[i],:]) @ self.Ar @ self.softmax(self.Cr[mask_r, :]).t() - self.brs[i][:, mask_r] - self.b2s[i][mask_2s[i],:]).pow(2).mean()

            
            if (self.Cg is not None) and (self.Cr is not None):
                loss4 = - torch.trace(self.softmax(self.Cg[mask_g,:]).t() @ self.A[np.ix_(mask_g, mask_r)] @ self.softmax(self.Cr[mask_r,:])) /torch.norm(self.softmax(self.Cg[mask_g,:])) / torch.norm(self.A[np.ix_(mask_g, mask_r)] @ self.softmax(self.Cr[mask_r,:]))
            
            if (alpha[6] != 0) and (self.Cg is not None):
                Corrg = self.softmax(self.Cg[mask_g,:]).t() @ self.softmax(self.Cg[mask_g,:])
                loss7 -= torch.norm(Corrg * self.mask_diagg) / torch.norm(Corrg) 
                
            if (alpha[6] != 0) and (self.Cr is not None):    
                Corrr = self.softmax(self.Cr[mask_r,:]).t() @ self.softmax(self.Cr[mask_r,:])
                loss7 -= torch.norm(Corrr * self.mask_diagr) / torch.norm(Corrr)

            
            
        elif mode == "A_g":
            
            for i, G in enumerate(self.Gs):
                self.C1s[i].requires_grad = False
                self.Cg.requires_grad = False
                self.Ag.requires_grad = True
                if self.Ar is not None:
                    self.Ar.requires_grad = False
                
                loss1 += (G[np.ix_(mask_1s[i], mask_g)] - self.softmax(self.C1s[i][mask_1s[i],:]) @ self.Ag @ self.softmax(self.Cg[mask_g,:]).t() - self.bgs[i][:, mask_g] - self.b1s[i][mask_1s[i],:]).pow(2).mean()
            
            if (self.Ag is not None) and (self.Ar is not None):
                loss3 = - torch.trace(self.Ar @ self.Ag.t())/torch.norm(self.Ar)/torch.norm(self.Ag)
                
            loss8 = self.Ag.abs().sum()

        
        elif mode == "A_r":

            for i, R in enumerate(self.Rs): 
                self.C2s[i].requires_grad = False
                self.Cr.requires_grad = False
                self.Ar.requires_grad = True
                if self.Ag is not None:
                    self.Ag.requires_grad = False
                
                loss2 += (R[np.ix_(mask_2s[i], mask_r)] - self.softmax(self.C2s[i][mask_2s[i],:]) @ self.Ar @ self.softmax(self.Cr[mask_r, :]).t() - self.brs[i][:, mask_r] - self.b2s[i][mask_2s[i],:]).pow(2).mean()

            if (self.Ag is not None) and (self.Ar is not None):                
                loss3 = - torch.trace(self.Ar @ self.Ag.t())/torch.norm(self.Ar)/torch.norm(self.Ag)
            
            loss8 = self.Ar.abs().sum()
            
            
        elif mode == "b":
            with torch.no_grad():
                for i, G in enumerate(self.Gs):
                    self.bgs[i][:, mask_g] = torch.mean(G[np.ix_(mask_1s[i], mask_g)] - self.softmax(self.C1s[i][mask_1s[i],:]) @ self.Ag @ self.softmax(self.Cg[mask_g, :]).t() - self.b1s[i][mask_1s[i], :], dim = 0)[None,:]
                    self.b1s[i][mask_1s[i], :] = torch.mean(G[np.ix_(mask_1s[i], mask_g)] - self.softmax(self.C1s[i][mask_1s[i],:]) @ self.Ag @ self.softmax(self.Cg[mask_g, :]).t() - self.bgs[i][:, mask_g], dim = 1)[:,None]
                
                for i, R in enumerate(self.Rs): 
                    self.brs[i][:, mask_r] = torch.mean(R[np.ix_(mask_2s[i], mask_r)] - self.softmax(self.C2s[i][mask_2s[i],:]) @ self.Ar @ self.softmax(self.Cr[mask_r, :]).t() - self.b2s[i][mask_2s[i],:], dim = 0)[None,:]
                    self.b2s[i][mask_2s[i], :] = torch.mean(R[np.ix_(mask_2s[i], mask_r)] - self.softmax(self.C2s[i][mask_2s[i],:]) @ self.Ar @ self.softmax(self.Cr[mask_r, :]).t() - self.brs[i][:, mask_r], dim = 1)[:,None]
            
                
        elif mode == 'valid':
            with torch.no_grad():
                
                for i, G in enumerate(self.Gs):
                    loss1 += (G - self.softmax(self.C1s[i]) @ self.Ag @ self.softmax(self.Cg).t() - self.bgs[i] - self.b1s[i]).pow(2).mean()
                    
                    if len(self.missing_dims[i]) > 0: 
                        loss5 += (self.softmax(self.C1s[i])[:,self.missing_dims[i]]).mean()
                        
                    if alpha[5] != 0:
                        Corr1 = self.softmax(self.C1s[i])[:,self.dims[i]].t() @ self.softmax(self.C1s[i])[:,self.dims[i]]
                        loss6 -= torch.norm(Corr1 * self.mask_diags[i]) / torch.norm(Corr1)
                
                
                for i, R in enumerate(self.Rs): 
                    loss2 += (R - self.softmax(self.C2s[i]) @ self.Ar @ self.softmax(self.Cr).t() - self.brs[i] - self.b2s[i]).pow(2).mean()
                    if len(self.missing_dims[i + len(self.Gs)]) > 0: 
                        loss5 += (self.softmax(self.C2s[i])[:,self.missing_dims[i + len(self.Gs)]]).mean()
                    
                    if alpha[5] != 0:
                        Corr2 = self.softmax(self.C2s[i])[:,self.dims[i + len(self.Gs)]].t() @ self.softmax(self.C2s[i])[:,self.dims[i + len(self.Gs)]]
                        loss6 -= torch.trace(Corr2 * self.mask_diags[i + len(self.Gs)]) / torch.norm(Corr2)
                    
                
            
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
            mask_1s = []
            mask_2s = []
            
            for i, G in enumerate(self.Gs):
                mask_1s.append(np.random.choice(self.Gs[0].shape[0], int(self.Gs[0].shape[0] * self.batch_size), replace=False))
            
            for i, R in enumerate(self.Rs): 
                mask_2s.append(np.random.choice(self.Rs[0].shape[0], int(self.Rs[0].shape[0] * self.batch_size), replace=False))
            
            if len(self.Gs) != 0:
                mask_g = np.random.choice(G.shape[1], int(G.shape[1] * self.batch_size), replace=False)
            else:
                mask_g = None
                
            if len(self.Rs) != 0:
                mask_r = np.random.choice(R.shape[1], int(R.shape[1] * self.batch_size), replace=False)
            else:
                mask_r = None
            
            # update alpha with self.alpha
            if t == T1:
                alpha = self.alpha
            
            for mode in self.orders:
                loss, *_ = self.batch_loss(mode, alpha, [mask_1s, mask_g, mask_2s, mask_r])
                
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
