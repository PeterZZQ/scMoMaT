import sys, os
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as opt
import pandas as pd
from torch.nn import Module, Parameter
from torch import softmax, log_softmax, Tensor
import itertools

from sklearn.cluster import KMeans
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import time
import utils

import matplotlib.pyplot as plt

from sparsemax import Sparsemax

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class cfrm_diff(Module):
    def __init__(self, counts, N1=3, N2 = 3, K = 3, batch_size=0.3, interval=10, lr=1e-3, alpha = [1000, 1000, 100, 100, 0.0, 0.0, 0.0], seed = None, learn_gact = False):
        super().__init__()
        self.N1 = N1
        self.N2 = N2
        self.K = K
        self.batch_size = batch_size
        self.interval = interval
        self.alpha = torch.FloatTensor(alpha).to(device)
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # data
        counts_rna = counts["rna"][0]
        counts_atac = counts["atac"][0]
        gact = counts["gact"][0]

        counts_rna = utils.preprocess(counts_rna, mode = "quantile", modality = "RNA")
        counts_atac = utils.preprocess(counts_atac, mode = "quantile", modality = "ATAC")

        # necessary, to force them into the same scale, make it easier for A.
        counts_rna = counts_rna/np.max(counts_rna)
        counts_atac = counts_atac/np.max(counts_atac)
        # sum of all regions within a gene to be one
        assert gact.shape[0] < gact.shape[1]
        gact = utils.preprocess(gact, mode = "gact")

        self.G = torch.FloatTensor(counts_rna).to(device)
        self.R = torch.FloatTensor(counts_atac).to(device)
        self.A = torch.FloatTensor(gact).to(device)

        # not always the same scale, only the latent dimension of K is shared
        self.N_feat = self.N1 + self.N2 - K
        Ainit = torch.rand((self.N_feat, self.N_feat))
        self.A_g = Parameter(Ainit)
        self.A_r = Parameter(Ainit)

        self.C_1 = Parameter(torch.rand(self.G.shape[0], self.N_feat))
        self.C_2 = Parameter(torch.rand(self.R.shape[0], self.N_feat))

        self.C_r = Parameter(torch.rand(self.R.shape[1], self.N_feat))
        self.C_g = Parameter(torch.rand(self.G.shape[1], self.N_feat))
        
        self.b_g = torch.zeros(1, self.G.shape[1]).to(device)
        self.b_r = torch.zeros(1, self.R.shape[1]).to(device)
        self.b_1 = torch.zeros(self.G.shape[0],1).to(device)
        self.b_2 = torch.zeros(self.R.shape[0],1).to(device)
        
        self.optimizer = opt.Adam(self.parameters(), lr=lr)
        self.learn_gact = learn_gact
        if self.learn_gact:
            self.B = Parameter((self.A > 0).float())
        else:
            self.B = (self.A > 0).float()        

    
    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim = 1)

    @staticmethod
    def sparsemax(X: Tensor):
        # sparsemax = Sparsemax(dim=1)
        # return sparsemax(X)
        return torch.softmax(X, dim = 1)
    
    def batch_loss(self, mode, alpha):
        dim1 = [x for x in range(self.N1)]
        dim2 = [x for x in range(self.K)] + [x for x in range(self.N1, self.N_feat)]

        if mode != 'valid':
            mask_1 = np.random.choice(self.G.shape[0], int(self.G.shape[0] * self.batch_size), replace=False)
            mask_2 = np.random.choice(self.R.shape[0], int(self.R.shape[0] * self.batch_size), replace=False)
            mask_g = np.random.choice(self.G.shape[1], int(self.G.shape[1] * self.batch_size), replace=False)
            mask_r = np.random.choice(self.R.shape[1], int(self.R.shape[1] * self.batch_size), replace=False)

        if mode == 'C_12':
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.sparsemax(self.C_1[mask_1,:]) @ self.A_g.detach() @ self.softmax(self.C_g[mask_g, :].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.sparsemax(self.C_2[mask_2,:]) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r,:].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()    
            
            loss3 = 0
            loss4 = 0
            
            # make corresponding dimensions approach 0
            loss5 = 0
            if self.N1 > self.K:
                # cluster of C2 that doesn't exist should be removed
                loss5 += (self.sparsemax(self.C_2[mask_2,:])[:,self.K:self.N1]).mean()
            if self.N2 > self.K:
                loss5 += (self.sparsemax(self.C_1[mask_1,:])[:,self.N1:self.N_feat]).mean()

            if alpha[5] != 0:
                Corr_1 = self.sparsemax(self.C_1[mask_1,:])[:,dim1].t() @ self.sparsemax(self.C_1[mask_1, :])[:,dim1]
                Corr_2 = self.sparsemax(self.C_2[mask_2,:])[:,dim2].t() @ self.sparsemax(self.C_2[mask_2, :])[:,dim2]
                # loss6 = - torch.trace(Corr_1) / torch.norm(Corr_1) - torch.trace(Corr_2) / torch.norm(Corr_2)
                loss6 = - torch.norm(Corr_1 * torch.eye(Corr_1.shape[0]).to(device)) / torch.norm(Corr_1) - torch.trace(Corr_2 * torch.eye(Corr_2.shape[0]).to(device)) / torch.norm(Corr_2)
            
            else:
                loss6 = 0
            
            loss7 = 0
            
        elif mode == 'C_gr':
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.sparsemax(self.C_1[mask_1,:].detach()) @ self.A_g.detach() @ self.softmax(self.C_g)[mask_g, :].t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.sparsemax(self.C_2[mask_2,:].detach()) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = 0 
            Aint = self.A[np.ix_(mask_g, mask_r)] * self.B[np.ix_(mask_g, mask_r)].detach()
            
            loss4 = - torch.trace(self.softmax(self.C_g[mask_g,:]).t() @ Aint @ self.softmax(self.C_r[mask_r,:])) /torch.norm(self.softmax(self.C_g[mask_g,:])) / torch.norm(Aint @ self.softmax(self.C_r[mask_r,:]))
            
            loss5 = 0
            loss6 = 0
            
            if alpha[6]!=0:
                Corr_g = self.softmax(self.C_g[mask_g,:]).t() @ self.softmax(self.C_g[mask_g,:])
                Corr_r = self.softmax(self.C_r[mask_r,:]).t() @ self.softmax(self.C_r[mask_r,:])
                # loss7 = - torch.trace(Corr_g) / torch.norm(Corr_g) - torch.trace(Corr_r) / torch.norm(Corr_r)
                loss7 = - torch.norm(Corr_g * torch.eye(Corr_g.shape[0]).to(device)) / torch.norm(Corr_g) - torch.norm(Corr_r * torch.eye(Corr_r.shape[0]).to(device)) / torch.norm(Corr_r)
            else:
                loss7 = 0
            
        
        elif mode == "Aint":
        
            loss1 = 0

            loss2 = 0

            loss3 = 0 
            Aint = self.A[np.ix_(mask_g, mask_r)] * self.B[np.ix_(mask_g, mask_r)]
            
            loss4 = - torch.trace(self.softmax(self.C_g[mask_g,:].detach()).t() @ Aint @ self.softmax(self.C_r[mask_r,:].detach())) /torch.norm(self.softmax(self.C_g[mask_g,:].detach())) / torch.norm(Aint @ self.softmax(self.C_r[mask_r,:].detach()))
            loss5 = 0
            loss6 = 0
            loss7 = 0
            
        
        elif mode == "A_g":
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.sparsemax(self.C_1[mask_1,:].detach()) @ self.A_g @  self.softmax(self.C_g[mask_g,:].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = 0

            # don't use [:self.K, :], because last dimension also need to match
            loss3 = - torch.trace(self.A_r.detach() @ self.A_g.t())/torch.norm(self.A_r.detach())/torch.norm(self.A_g)
            loss4 = 0
            loss5 = 0
            loss6 = 0
            loss7 = 0
        
        elif mode == "A_r":
            loss1 = 0
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.sparsemax(self.C_2[mask_2,:].detach()) @ self.A_r @ self.softmax(self.C_r[mask_r, :].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            # don't use [:self.K, :], because last dimension also need to match
            loss3 = - torch.trace(self.A_r @ self.A_g.detach().t())/torch.norm(self.A_r)/torch.norm(self.A_g.detach())
            loss4 = 0
            loss5 = 0
            loss6 = 0
            loss7 = 0
       
        elif mode == "b":
            with torch.no_grad():
                self.b_g[:, mask_g] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.sparsemax(self.C_1[mask_1,:]) @ self.A_g @ self.softmax(self.C_g)[mask_g,:].t() - self.b_1[mask_1,:], dim = 0)[None,:]/mask_1.shape[0]
                self.b_r[:, mask_r] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.sparsemax(self.C_2[mask_2,:]) @ self.A_r @ self.softmax(self.C_r[mask_r, :]).t() - self.b_2[mask_2,:], dim = 0)[None,:]/mask_2.shape[0]
                self.b_1[mask_1, :] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.sparsemax(self.C_1[mask_1,:]) @ self.A_g @  self.softmax(self.C_g)[mask_g,:].t() - self.b_g[:, mask_g], dim = 1)[:,None]/mask_g.shape[0]
                self.b_2[mask_2, :] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.sparsemax(self.C_2[mask_2,:]) @ self.A_r @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r], dim = 1)[:,None]/mask_r.shape[0]
            
                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0
                loss5 = 0
                loss6 = 0
                loss7 = 0
                
        elif mode == 'valid':
            with torch.no_grad():
                loss1 = (self.G - self.sparsemax(self.C_1) @ self.A_g @ self.softmax(self.C_g).t() - self.b_g - self.b_1).pow(2).mean()
                     
                loss2 = (self.R - self.sparsemax(self.C_2) @ self.A_r @ self.softmax(self.C_r).t() - self.b_r - self.b_2).pow(2).mean()
                 
                loss3 = - torch.trace(self.A_r[:self.K, :] @ self.A_g[:self.K, :].t())/torch.norm(self.A_r[:self.K, :])/torch.norm(self.A_g[:self.K, :])
                
                Aint = self.A * self.B
                loss4 = - torch.trace(self.softmax(self.C_g).t() @ Aint @ self.softmax(self.C_r)) /torch.norm(self.softmax(self.C_g)) / torch.norm(Aint @ self.softmax(self.C_r))
                
                loss6 = 0
                if alpha[5] != 0:
                    Corr_1 = self.sparsemax(self.C_1)[:,dim1].t() @ self.sparsemax(self.C_1)[:,dim1]
                    Corr_2 = self.sparsemax(self.C_2)[:,dim2].t() @ self.sparsemax(self.C_2)[:,dim2]
                    # loss6 = - torch.trace(Corr_1) / torch.norm(Corr_1) - torch.trace(Corr_2) / torch.norm(Corr_2)
                    loss6 = - torch.norm(Corr_1 * torch.eye(Corr_1.shape[0]).to(device)) / torch.norm(Corr_1) - torch.trace(Corr_2 * torch.eye(Corr_2.shape[0]).to(device)) / torch.norm(Corr_2)
                
                loss7 = 0
                if alpha[6] != 0:
                    Corr_g = self.softmax(self.C_g).t() @ self.softmax(self.C_g)
                    Corr_r = self.softmax(self.C_r).t() @ self.softmax(self.C_r)
                    # loss7 = - torch.trace(Corr_g) / torch.norm(Corr_g) - torch.trace(Corr_r) / torch.norm(Corr_r)
                    loss7 = - torch.norm(Corr_g * torch.eye(Corr_g.shape[0]).to(device)) / torch.norm(Corr_g) - torch.norm(Corr_r * torch.eye(Corr_r.shape[0]).to(device)) / torch.norm(Corr_r)
                
                loss5 = 0
                if self.N1 > self.K:
                    loss5 += (self.sparsemax(self.C_2)[:,self.K:self.N1]).mean()
                if self.N2 > self.K:
                    loss5 += (self.sparsemax(self.C_1)[:,self.N1:self.N_feat]).mean()
        else:
            raise NotImplementedError
        loss = alpha[0] * loss1 + alpha[1] * loss2 + alpha[2] * loss3 + alpha[3] * loss4 + alpha[4] * loss5 + alpha[5] * loss6 + alpha[6] * loss7
        
        return loss, alpha[0] * loss1, alpha[1] * loss2, alpha[2] * loss3, alpha[3] * loss4, alpha[4] * loss5, alpha[5] * loss6, alpha[6] * loss7  
    
    def train_func(self, T):
        best_loss = 1e12
        count = 0

        T1 = int(T/10)

        if self.learn_gact:
            orders = ['b']* 1 + ['C_12']*1 + ['C_gr']*1 + ['Aint']*1 + ['A_g']*1 + ['A_r']*1
        else:
            orders = ['b']* 1 + ['C_12']*1 + ['C_gr']*1 + ['A_g']*1 + ['A_r']*1
            
        alpha = torch.zeros_like(self.alpha).to(device)
        alpha[:5] = self.alpha[:5]

        for t in range(T1):

            for mode in orders: 
                loss, *_ = self.batch_loss(mode, alpha)
                
                if mode != 'b':
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                
            if (t+1) % self.interval == 0:
                
                loss, loss1, loss2, loss3, loss4, loss5, loss6, loss7 = self.batch_loss('valid', alpha)
                
                print('Epoch {}, Validating Loss: {:.4f}'.format(t + 1, loss.item()))
                info = [
                    'loss 1: {:.5f}'.format(loss1.item()),
                    'loss 2: {:.5f}'.format(loss2.item()),
                    'loss 3: {:.5f}'.format(loss3.item()),
                    'loss 4: {:.5f}'.format(loss4.item()),
                    'loss 5: {:.5f}'.format(loss5.item()),
                    'loss 6: {:.5f}'.format(loss6.item()),
                    'loss 7: {:.5f}'.format(loss7.item()),
                ]
                for i in info:
                    print("\t", i)

        
        for t in range(T1, T):

            for mode in orders:
                loss, *_ = self.batch_loss(mode, alpha)
                
                if mode != 'b':
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                
            if (t+1) % self.interval == 0:
                loss, loss1, loss2, loss3, loss4, loss5, loss6, loss7 = self.batch_loss('valid', self.alpha)
                
                print('Epoch {}, Validating Loss: {:.4f}'.format(t + 1, loss.item()))
                info = [
                    'loss 1: {:.5f}'.format(loss1.item()),
                    'loss 2: {:.5f}'.format(loss2.item()),
                    'loss 3: {:.5f}'.format(loss3.item()),
                    'loss 4: {:.5f}'.format(loss4.item()),
                    'loss 5: {:.5f}'.format(loss5.item()),
                    'loss 6: {:.5f}'.format(loss6.item()),
                    'loss 7: {:.5f}'.format(loss7.item()),
                ]
                for i in info:
                    print("\t", i)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(self.state_dict(), f'../check_points/real_{self.N1}.pt')
                    count = 0
                else:
                    count += 1
                    if count % 40 == 0:
                        self.optimizer.param_groups[0]['lr'] *= 0.5
                        print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < 1e-6:
                            break
                        else:
                            self.load_state_dict(torch.load(f'../check_points/real_{self.N1}.pt'))
                            count = 0
                            

                            
                            
"""
class cfrm_diff(Module):
    def __init__(self, counts, N1=3, N2 = 3, K = 3, batch_size=0.3, interval=10, lr=1e-3, alpha = [1000, 1000, 100, 0.0, 0.0], seed = None, learn_gact = False):
        super().__init__()
        self.N1 = N1
        self.N2 = N2
        self.K = K
        self.batch_size = batch_size
        self.interval = interval
        self.alpha = torch.FloatTensor(alpha).to(device)
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # data
        counts_rna = counts["rna"][0]
        counts_atac = counts["atac"][0]
        gact = counts["gact"][0]

        counts_rna = utils.preprocess(counts_rna, mode = "quantile", modality = "RNA")
        counts_atac = utils.preprocess(counts_atac, mode = "quantile", modality = "ATAC")

        # necessary, to force them into the same scale, make it easier for A.
        counts_rna = counts_rna/np.max(counts_rna)
        counts_atac = counts_atac/np.max(counts_atac)
        # sum of all regions within a gene to be one
        assert gact.shape[0] < gact.shape[1]
        gact = utils.preprocess(gact, mode = "gact")

        self.G = torch.FloatTensor(counts_rna).to(device)
        self.R = torch.FloatTensor(counts_atac).to(device)
        self.A = torch.FloatTensor(gact).to(device)

        # not always the same scale
        self.N_feat = max(self.N1, self.N2)
        Ainit = torch.rand((K, self.N_feat))
        self.A_g = Parameter(torch.cat((Ainit, torch.rand(self.N1 - self.K, self.N_feat)), dim = 0))        
        self.A_r = Parameter(torch.cat((Ainit, torch.rand(self.N2 - self.K, self.N_feat)), dim = 0))

        self.C_1 = Parameter(torch.rand(self.G.shape[0], self.N1))
        self.C_2 = Parameter(torch.rand(self.R.shape[0], self.N2))
        # C_g is the average of C_r
        self.C_r = Parameter(torch.rand(self.R.shape[1], self.N_feat))
        self.C_g = Parameter(torch.rand(self.G.shape[1], self.N_feat))
        
        self.b_g = torch.zeros(1, self.G.shape[1]).to(device)
        self.b_r = torch.zeros(1, self.R.shape[1]).to(device)
        self.b_1 = torch.zeros(self.G.shape[0],1).to(device)
        self.b_2 = torch.zeros(self.R.shape[0],1).to(device)
        
        self.optimizer = opt.Adam(self.parameters(), lr=lr)
        self.learn_gact = learn_gact
        

    
    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim = 1)

    @staticmethod
    def sparsemax(X: Tensor):
        # sparsemax = Sparsemax(dim=1)
        # return sparsemax(X)
        return torch.softmax(X, dim = 1)
    
    def batch_loss(self, mode='C_c'):
        if mode != 'valid':
            mask_1 = np.random.choice(self.G.shape[0], int(self.G.shape[0] * self.batch_size), replace=False)
            mask_2 = np.random.choice(self.R.shape[0], int(self.R.shape[0] * self.batch_size), replace=False)
            mask_g = np.random.choice(self.G.shape[1], int(self.G.shape[1] * self.batch_size), replace=False)
            mask_r = np.random.choice(self.R.shape[1], int(self.R.shape[1] * self.batch_size), replace=False)
            
            
        if mode == 'C_12':
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.sparsemax(self.C_1[mask_1,:]) @ self.A_g.detach() @ self.softmax(self.C_g[mask_g, :].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.sparsemax(self.C_2[mask_2,:]) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r,:].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()    
            
            loss3 = 0
            loss4 = 0
            loss5 = - torch.trace(self.sparsemax(self.C_1[mask_1,:]).t() @ self.sparsemax(self.C_1[mask_1, :])) / torch.norm(self.sparsemax(self.C_1[mask_1,:]).t() @ self.sparsemax(self.C_1[mask_1, :])) \
                - torch.trace(self.sparsemax(self.C_2[mask_2,:]).t() @ self.sparsemax(self.C_2[mask_2, :])) / torch.norm(self.sparsemax(self.C_2[mask_2,:]).t() @ self.sparsemax(self.C_2[mask_2, :]))
            
                
        elif mode == 'C_gr':
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.sparsemax(self.C_1[mask_1,:].detach()) @ self.A_g.detach() @ self.softmax(self.C_g)[mask_g, :].t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.sparsemax(self.C_2[mask_2,:].detach()) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = 0 
            loss4 = - torch.trace(self.softmax(self.C_g[mask_g,:]).t() @ self.A[np.ix_(mask_g, mask_r)] @ self.softmax(self.C_r[mask_r,:])) /torch.norm(self.softmax(self.C_g[mask_g,:])) / torch.norm(self.A[np.ix_(mask_g, mask_r)] @ self.softmax(self.C_r[mask_r,:]))
            loss5 = - torch.trace(self.softmax(self.C_g[mask_g,:]).t() @ self.softmax(self.C_g[mask_g,:])) / torch.norm(self.softmax(self.C_g[mask_g,:]).t() @ self.softmax(self.C_g[mask_g,:])) - torch.trace(self.softmax(self.C_r[mask_r,:]).t() @ self.softmax(self.C_r[mask_r,:])) / torch.norm(self.softmax(self.C_r[mask_r,:]).t() @ self.softmax(self.C_r[mask_r,:]))
            

        elif mode == "A":
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.sparsemax(self.C_1[mask_1,:].detach()) @ self.A_g @  self.softmax(self.C_g[mask_g,:].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.sparsemax(self.C_2[mask_2,:].detach()) @ self.A_r @ self.softmax(self.C_r[mask_r, :].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = - torch.trace(self.A_r[:self.K, :] @ self.A_g[:self.K, :].t())/torch.norm(self.A_r[:self.K, :])/torch.norm(self.A_g[:self.K, :])
            loss4 = 0
            loss5 = 0
       
        elif mode == "b":
            with torch.no_grad():
                self.b_g[:, mask_g] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.sparsemax(self.C_1[mask_1,:]) @ self.A_g @ self.softmax(self.C_g)[mask_g,:].t() - self.b_1[mask_1,:], dim = 0)[None,:]/mask_1.shape[0]
                self.b_r[:, mask_r] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.sparsemax(self.C_2[mask_2,:]) @ self.A_r @ self.softmax(self.C_r[mask_r, :]).t() - self.b_2[mask_2,:], dim = 0)[None,:]/mask_2.shape[0]
                self.b_1[mask_1, :] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.sparsemax(self.C_1[mask_1,:]) @ self.A_g @  self.softmax(self.C_g)[mask_g,:].t() - self.b_g[:, mask_g], dim = 1)[:,None]/mask_g.shape[0]
                self.b_2[mask_2, :] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.sparsemax(self.C_2[mask_2,:]) @ self.A_r @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r], dim = 1)[:,None]/mask_r.shape[0]
            
                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0
                loss5 = 0
                
        elif mode == 'valid':
            with torch.no_grad():
                loss1 = (self.G - self.sparsemax(self.C_1) @ self.A_g @ self.softmax(self.C_g).t() - self.b_g - self.b_1).pow(2).mean()
                     
                loss2 = (self.R - self.sparsemax(self.C_2) @ self.A_r @ self.softmax(self.C_r).t() - self.b_r - self.b_2).pow(2).mean()
                 
                loss3 = - torch.trace(self.A_r[:self.K, :] @ self.A_g[:self.K, :].t())/torch.norm(self.A_r[:self.K, :])/torch.norm(self.A_g[:self.K, :])
                loss4 = - torch.trace(self.softmax(self.C_g).t() @ self.A @ self.softmax(self.C_r)) /torch.norm(self.softmax(self.C_g)) / torch.norm(self.A @ self.softmax(self.C_r))
                loss5 = - torch.trace(self.sparsemax(self.C_1).t() @ self.sparsemax(self.C_1)) / torch.norm(self.sparsemax(self.C_1).t() @ self.sparsemax(self.C_1)) - torch.trace(self.sparsemax(self.C_2).t() @ self.sparsemax(self.C_2)) / torch.norm(self.sparsemax(self.C_2).t() @ self.sparsemax(self.C_2))\
                    - torch.trace(self.softmax(self.C_g).t() @ self.softmax(self.C_g)) / torch.norm(self.softmax(self.C_g).t() @ self.softmax(self.C_g)) - torch.trace(self.softmax(self.C_r).t() @ self.softmax(self.C_r)) / torch.norm(self.softmax(self.C_r).t() @ self.softmax(self.C_r))
            
        
        else:
            raise NotImplementedError
        loss = self.alpha[0] * loss1 + self.alpha[1] * loss2 + self.alpha[2] * loss3 + self.alpha[3] * loss4 + self.alpha[4] * loss5
        
        return loss, self.alpha[0] * loss1, self.alpha[1] * loss2, self.alpha[2] * loss3, self.alpha[3] * loss4, self.alpha[4] * loss5   
    
    def train_func(self, T):
        best_loss = 1e12
        count = 0

        T1 = 2000 # int(T/4)

        alpha_temp = self.alpha[4].item()
        self.alpha[4] = 0 

        for t in range(T1):

            for mode in ['b']* 1 + ['C_12']*1 + ['C_gr']*1 + ['A']*1:
                loss, *_ = self.batch_loss(mode)
                
                if mode != 'b':
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                
            if (t+1) % self.interval == 0:
                loss, loss1, loss2, loss3, loss4, loss5 = self.batch_loss('valid')
                
                print('Epoch {}, Validating Loss: {:.4f}'.format(t + 1, loss.item()))
                info = [
                    'loss 1: {:.5f}'.format(loss1.item()),
                    'loss 2: {:.5f}'.format(loss2.item()),
                    'loss 3: {:.5f}'.format(loss3.item()),
                    'loss 4: {:.5f}'.format(loss4.item()),
                    'loss 5: {:.5f}'.format(loss5.item()),
                ]
                for i in info:
                    print("\t", i)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(self.state_dict(), f'../check_points/real_{self.N1}.pt')
                    count = 0
                else:
                    count += 1
                    if count % 40 == 0:
                        self.optimizer.param_groups[0]['lr'] *= 0.5
                        print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < 1e-6:
                            break
                        else:
                            self.load_state_dict(torch.load(f'../check_points/real_{self.N1}.pt'))
                            count = 0

        self.alpha[4] = alpha_temp

        print(self.alpha[4].item())

        for t in range(T1, T):

            for mode in ['b']* 1 + ['C_12']*1 + ['C_gr']*1 + ['A']*1:
                loss, *_ = self.batch_loss(mode)
                
                if mode != 'b':
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                
            if (t+1) % self.interval == 0:
                loss, loss1, loss2, loss3, loss4, loss5 = self.batch_loss('valid')
                
                print('Epoch {}, Validating Loss: {:.4f}'.format(t + 1, loss.item()))
                info = [
                    'loss 1: {:.5f}'.format(loss1.item()),
                    'loss 2: {:.5f}'.format(loss2.item()),
                    'loss 3: {:.5f}'.format(loss3.item()),
                    'loss 4: {:.5f}'.format(loss4.item()),
                    'loss 5: {:.5f}'.format(loss5.item()),
                ]
                for i in info:
                    print("\t", i)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(self.state_dict(), f'../check_points/real_{self.N1}.pt')
                    count = 0
                else:
                    count += 1
                    if count % 40 == 0:
                        self.optimizer.param_groups[0]['lr'] *= 0.5
                        print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < 1e-6:
                            break
                        else:
                            self.load_state_dict(torch.load(f'../check_points/real_{self.N1}.pt'))
                            count = 0

"""