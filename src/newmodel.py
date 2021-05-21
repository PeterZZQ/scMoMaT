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
from sparsemax import Sparsemax

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class cfrm_diff(Module):
    def __init__(self, counts, N1 = 3, N2 = 3, K = 3, batch_size=1.0, interval=10, lr=1e-3, alpha = [1000, 1000, 100, 100, 0.0], seed = None, init = None, learn_gact = False):
        super().__init__()
        
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
        
        if learn_gact:
            self.B = Parameter(torch.ones(self.A.shape))
            self.opt_order = ['b']* 1 + ['C_12']*1 + ['C_gr']*1 + ['A']*1 + ['B']*1
            
        else:
            self.B = torch.ones(self.A.shape).to(device)
            self.opt_order = ['b']* 1 + ['C_12']*1 + ['C_gr']*1 + ['A']*1
            self.alpha[4] = 0
            
        self.N1 = N1
        self.N2 = N2
        self.K = K

        # not always the same scale
        if init is None:
            # how to specify N_feat??
            N_feat = max(self.N1, self.N2)
            Aint = torch.randn((self.K, N_feat))
            self.A_g = Parameter(torch.cat((Aint,torch.rand(self.N1 - self.K, N_feat)), dim = 0))
            self.A_r = Parameter(torch.cat((Aint,torch.rand(self.N2 - self.K, N_feat)), dim = 0))

            self.C_1 = Parameter(torch.rand(self.G.shape[0], self.N1))
            self.C_2 = Parameter(torch.rand(self.R.shape[0], self.N2))
            
            self.C_r = Parameter(torch.rand(self.R.shape[1], N_feat))
            self.C_g = Parameter(torch.rand(self.G.shape[1], N_feat))
            
        
        else:
            C_1, C_2, A_g, A_r, C_g, C_r = init
            self.C_1 = Parameter(C_1)
            self.C_2 = Parameter(C_2)
            self.C_g = Parameter(C_g)
            self.C_r = Parameter(C_r)
            self.A_r = Parameter(A_r)
            self.A_g = Parameter(A_g)
        
        self.b_g = torch.zeros(1, self.G.shape[1]).to(device)
        self.b_r = torch.zeros(1, self.R.shape[1]).to(device)
        self.b_1 = torch.zeros(self.G.shape[0],1).to(device)
        self.b_2 = torch.zeros(self.R.shape[0],1).to(device)
        
        self.optimizer = opt.Adam(self.parameters(), lr=lr)
        
        
        

    
    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim = 1)
        # sparsemax = Sparsemax(dim=1)
        # return sparsemax(X)
    
    @staticmethod
    def entropy_loss(C):
        loss = - (F.softmax(C, dim=1) - 0.5).pow(2).mean()
        
        return loss
    
    def batch_loss(self, mode='C_c'):
        if mode != 'valid':
            mask_1 = np.random.choice(self.G.shape[0], int(self.G.shape[0] * self.batch_size), replace=False)
            mask_2 = np.random.choice(self.R.shape[0], int(self.R.shape[0] * self.batch_size), replace=False)
            mask_g = np.random.choice(self.G.shape[1], int(self.G.shape[1] * self.batch_size), replace=False)
            mask_r = np.random.choice(self.R.shape[1], int(self.R.shape[1] * self.batch_size), replace=False)
            
            
        if mode == 'C_12':
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.A_g.detach() @ self.softmax(self.C_g[mask_g, :].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r,:].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()    
            
            loss3 = 0
            loss4 = 0 
            loss5 = 0
            
                
        elif mode == 'C_gr':
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g.detach() @ self.softmax(self.C_g)[mask_g, :].t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = 0 
            loss4 = - torch.trace(self.softmax(self.C_g[mask_g,:]).t() @ (self.A[np.ix_(mask_g, mask_r)] * self.B[np.ix_(mask_g, mask_r)].detach()) @ self.softmax(self.C_r[mask_r,:])) /torch.norm(self.softmax(self.C_g[mask_g,:])) / torch.norm((self.A[np.ix_(mask_g, mask_r)] * self.B[np.ix_(mask_g, mask_r)].detach()) @ self.softmax(self.C_r[mask_r,:]))
            loss5 = 0

        elif mode == 'B':
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g.detach() @ self.softmax(self.C_g.detach())[mask_g, :].t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r, :].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = 0
            
            Aint = self.A[np.ix_(mask_g, mask_r)] * self.B[np.ix_(mask_g, mask_r)]
            
            loss4 = - torch.trace(self.softmax(self.C_g[mask_g,:].detach()).t() @ Aint @ self.softmax(self.C_r[mask_r,:].detach())) / torch.norm(self.softmax(self.C_g[mask_g,:].detach())) / torch.norm(Aint @ self.softmax(self.C_r[mask_r,:].detach()))
            
            loss5 = Aint.abs().sum()/torch.sum(self.A)
            
 
          
        elif mode == "A":
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g @  self.softmax(self.C_g[mask_g,:].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r @ self.softmax(self.C_r[mask_r, :].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = - torch.trace(self.A_g[:self.K,:] @ self.A_r[:self.K,:].t())/torch.norm(self.A_r[:self.K,:])/torch.norm(self.A_g[:self.K,:])
            loss4 = 0
            loss5 = 0
       
        elif mode == "b":
            with torch.no_grad():
                self.b_g[:, mask_g] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.A_g @ self.softmax(self.C_g)[mask_g,:].t() - self.b_1[mask_1,:], dim = 0)[None,:]/mask_1.shape[0]
                self.b_r[:, mask_r] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ self.A_r @ self.softmax(self.C_r[mask_r, :]).t() - self.b_2[mask_2,:], dim = 0)[None,:]/mask_2.shape[0]
                self.b_1[mask_1, :] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.A_g @  self.softmax(self.C_g)[mask_g,:].t() - self.b_g[:, mask_g], dim = 1)[:,None]/mask_g.shape[0]
                self.b_2[mask_2, :] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ self.A_r @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r], dim = 1)[:,None]/mask_r.shape[0]
            
                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0
                loss5 = 0
                
        elif mode == 'valid':
            with torch.no_grad():
                loss1 = (self.G - self.softmax(self.C_1) @ self.A_g @ self.softmax(self.C_g).t() - self.b_g - self.b_1).pow(2).mean()
                     
                loss2 = (self.R - self.softmax(self.C_2) @ self.A_r @ self.softmax(self.C_r).t() - self.b_r - self.b_2).pow(2).mean()
                 
                loss3 = - torch.trace(self.A_g[:self.K,:] @ self.A_r[:self.K,:].t())/torch.norm(self.A_r[:self.K,:])/torch.norm(self.A_g[:self.K,:])
                
                Aint = self.A * self.B
                loss4 = - torch.trace(self.softmax(self.C_g).t() @ Aint @ self.softmax(self.C_r)) /torch.norm(self.softmax(self.C_g)) / torch.norm(Aint @ self.softmax(self.C_r)) 
                
                loss5 = Aint.abs().sum()/torch.sum(self.A)
                     
        else:
            raise NotImplementedError
        loss = self.alpha[0] * loss1 + self.alpha[1] * loss2 + self.alpha[2] * loss3 + self.alpha[3] * loss4 + self.alpha[4] * loss5
        
        return loss, self.alpha[0] * loss1, self.alpha[1] * loss2, self.alpha[2] * loss3, self.alpha[3] * loss4, self.alpha[4] * loss5   
    
    def train_func(self, T):
        best_loss = 1e12
        count = 0
        
        for t in range(T):
            
            for mode in self.opt_order:
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
                    'loss 5: {:.5f}'.format(loss5.item())
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



######################################################################################################################            
class cfrm_new(Module):
    def __init__(self, counts, N=3, batch_size=1.0, interval=10, lr=1e-3, alpha = [1000, 1000, 100, 100, 0.0], seed = None, init = None):
        super().__init__()
        self.N1 = N
        self.N2 = N
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
        self.B = Parameter(torch.ones(self.A.shape))

        # not always the same scale
        if init is None:
            Ainit = torch.randn((self.N1,self.N2))
            self.A_r = Parameter(Ainit)        
            self.A_g = Parameter(Ainit)

            self.C_1 = Parameter(torch.rand(self.G.shape[0], self.N1))
            self.C_2 = Parameter(torch.rand(self.R.shape[0], self.N1))
            # C_g is the average of C_r
            self.C_r = Parameter(torch.rand(self.R.shape[1], self.N2))
            self.C_g = Parameter(torch.rand(self.G.shape[1], self.N2))
            
        
        else:
            C_1, C_2, A_g, A_r, C_g, C_r = init
            self.C_1 = Parameter(C_1)
            self.C_2 = Parameter(C_2)
            self.C_g = Parameter(C_g)
            self.C_r = Parameter(C_r)
            self.A_r = Parameter(A_r)
            self.A_g = Parameter(A_g)
        
        self.b_g = torch.zeros(1, self.G.shape[1]).to(device)
        self.b_r = torch.zeros(1, self.R.shape[1]).to(device)
        self.b_1 = torch.zeros(self.G.shape[0],1).to(device)
        self.b_2 = torch.zeros(self.R.shape[0],1).to(device)
        
        self.optimizer = opt.Adam(self.parameters(), lr=lr)
        

    
    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim = 1)
    
    @staticmethod
    def entropy_loss(C):
        loss = - (F.softmax(C, dim=1) - 0.5).pow(2).mean()
        
        return loss
    
    def batch_loss(self, mode='C_c'):
        if mode != 'valid':
            mask_1 = np.random.choice(self.G.shape[0], int(self.G.shape[0] * self.batch_size), replace=False)
            mask_2 = np.random.choice(self.R.shape[0], int(self.R.shape[0] * self.batch_size), replace=False)
            mask_g = np.random.choice(self.G.shape[1], int(self.G.shape[1] * self.batch_size), replace=False)
            mask_r = np.random.choice(self.R.shape[1], int(self.R.shape[1] * self.batch_size), replace=False)
            
            
        if mode == 'C_12':
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.A_g.detach() @ self.softmax(self.C_g[mask_g, :].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r,:].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()    
            
            loss3 = 0
            loss4 = 0 
            loss5 = 0
            
                
        elif mode == 'C_gr':
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g.detach() @ self.softmax(self.C_g)[mask_g, :].t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = 0 
            loss4 = - torch.trace(self.softmax(self.C_g[mask_g,:]).t() @ (self.A[np.ix_(mask_g, mask_r)] * self.B[np.ix_(mask_g, mask_r)].detach()) @ self.softmax(self.C_r[mask_r,:])) /torch.norm(self.softmax(self.C_g[mask_g,:])) / torch.norm((self.A[np.ix_(mask_g, mask_r)] * self.B[np.ix_(mask_g, mask_r)].detach()) @ self.softmax(self.C_r[mask_r,:]))
            loss5 = 0

        elif mode == 'B':
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g.detach() @ self.softmax(self.C_g.detach())[mask_g, :].t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r, :].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = 0
            
            Aint = self.A[np.ix_(mask_g, mask_r)] * self.B[np.ix_(mask_g, mask_r)]
            
            loss4 = - torch.trace(self.softmax(self.C_g[mask_g,:].detach()).t() @ Aint @ self.softmax(self.C_r[mask_r,:].detach())) / torch.norm(self.softmax(self.C_g[mask_g,:].detach())) / torch.norm(Aint @ self.softmax(self.C_r[mask_r,:].detach()))
            
            loss5 = Aint.abs().sum()/torch.sum(self.A)
            
 
          
        elif mode == "A":
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g @  self.softmax(self.C_g[mask_g,:].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r @ self.softmax(self.C_r[mask_r, :].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = - torch.trace(self.A_r.t() @ self.A_g)/torch.norm(self.A_r)/torch.norm(self.A_g)
            loss4 = 0
            loss5 = 0
       
        elif mode == "b":
            with torch.no_grad():
                self.b_g[:, mask_g] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.A_g @ self.softmax(self.C_g)[mask_g,:].t() - self.b_1[mask_1,:], dim = 0)[None,:]/mask_1.shape[0]
                self.b_r[:, mask_r] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ self.A_r @ self.softmax(self.C_r[mask_r, :]).t() - self.b_2[mask_2,:], dim = 0)[None,:]/mask_2.shape[0]
                self.b_1[mask_1, :] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.A_g @  self.softmax(self.C_g)[mask_g,:].t() - self.b_g[:, mask_g], dim = 1)[:,None]/mask_g.shape[0]
                self.b_2[mask_2, :] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ self.A_r @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r], dim = 1)[:,None]/mask_r.shape[0]
            
                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0
                loss5 = 0
                
        elif mode == 'valid':
            with torch.no_grad():
                loss1 = (self.G - self.softmax(self.C_1) @ self.A_g @ self.softmax(self.C_g).t() - self.b_g - self.b_1).pow(2).mean()
                     
                loss2 = (self.R - self.softmax(self.C_2) @ self.A_r @ self.softmax(self.C_r).t() - self.b_r - self.b_2).pow(2).mean()
                 
                loss3 = - torch.trace(self.A_r.t() @ self.A_g)/torch.norm(self.A_r)/torch.norm(self.A_g)
                Aint = self.A * self.B
                loss4 = - torch.trace(self.softmax(self.C_g).t() @ Aint @ self.softmax(self.C_r)) /torch.norm(self.softmax(self.C_g)) / torch.norm(Aint @ self.softmax(self.C_r)) 
                loss5 = Aint.abs().sum()/torch.sum(self.A)
                     
        else:
            raise NotImplementedError
        loss = self.alpha[0] * loss1 + self.alpha[1] * loss2 + self.alpha[2] * loss3 + self.alpha[3] * loss4 + self.alpha[4] * loss5
        
        return loss, self.alpha[0] * loss1, self.alpha[1] * loss2, self.alpha[2] * loss3, self.alpha[3] * loss4, self.alpha[4] * loss5   
    
    def train_func(self, T):
        best_loss = 1e12
        count = 0
        
        for t in range(T):
            
            for mode in ['b']* 1 + ['C_12']*1 + ['C_gr']*1 + ['A']*1 + ['B']*1:
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
                    'loss 5: {:.5f}'.format(loss5.item())
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



class cfrm_new2(Module):
    def __init__(self, counts, N=3, batch_size=1.0, interval=10, lr=1e-3, alpha = [1000, 1000, 100, 100, 0.0], seed = None, init = None):
        super().__init__()
        self.N1 = N
        self.N2 = N
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
        self.B = Parameter(torch.ones(self.A.shape))

        # not always the same scale
        if init is None:
            Ainit = torch.randn((self.N1,self.N2))
            self.A_r = Parameter(Ainit)        
            self.A_g = Parameter(Ainit)

            self.C_1 = Parameter(torch.rand(self.G.shape[0], self.N1))
            self.C_2 = Parameter(torch.rand(self.R.shape[0], self.N1))
            # C_g is the average of C_r
            self.C_r = Parameter(torch.rand(self.R.shape[1], self.N2))
            self.C_g = Parameter(torch.rand(self.G.shape[1], self.N2))
            
        
        else:
            C_1, C_2, A_g, A_r, C_g, C_r = init
            self.C_1 = Parameter(C_1)
            self.C_2 = Parameter(C_2)
            self.C_g = Parameter(C_g)
            self.C_r = Parameter(C_r)
            self.A_r = Parameter(A_r)
            self.A_g = Parameter(A_g)
        
        self.b_g = torch.zeros(1, self.G.shape[1]).to(device)
        self.b_r = torch.zeros(1, self.R.shape[1]).to(device)
        self.b_1 = torch.zeros(self.G.shape[0],1).to(device)
        self.b_2 = torch.zeros(self.R.shape[0],1).to(device)
        
        self.optimizer = opt.Adam(self.parameters(), lr=lr)
        

    
    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim = 1)
    
    @staticmethod
    def entropy_loss(C):
        loss = - (F.softmax(C, dim=1) - 0.5).pow(2).mean()
        
        return loss
    
    def batch_loss(self, mode='C_c'):
        if mode != 'valid':
            mask_1 = np.random.choice(self.G.shape[0], int(self.G.shape[0] * self.batch_size), replace=False)
            mask_2 = np.random.choice(self.R.shape[0], int(self.R.shape[0] * self.batch_size), replace=False)
            mask_g = np.random.choice(self.G.shape[1], int(self.G.shape[1] * self.batch_size), replace=False)
            mask_r = np.random.choice(self.R.shape[1], int(self.R.shape[1] * self.batch_size), replace=False)
            
            
        if mode == 'C_12':
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.A_g.detach() @ self.softmax(self.C_g[mask_g, :].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r,:].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()    
            
            loss3 = 0
            loss4 = 0 
            loss5 = 0
            
                
        elif mode == 'C_gr':
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g.detach() @ self.softmax(self.C_g)[mask_g, :].t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = 0 
            loss4 = - torch.trace(self.softmax(self.C_g[mask_g,:]).t() @ (self.A[np.ix_(mask_g, mask_r)] * self.B[np.ix_(mask_g, mask_r)].detach()) @ self.softmax(self.C_r[mask_r,:])) /torch.norm(self.softmax(self.C_g[mask_g,:])) / torch.norm((self.A[np.ix_(mask_g, mask_r)] * self.B[np.ix_(mask_g, mask_r)].detach()) @ self.softmax(self.C_r[mask_r,:]))
            loss5 = 0

        elif mode == 'B':
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g.detach() @ self.softmax(self.C_g.detach())[mask_g, :].t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r, :].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = 0
            
            Aint = self.A[np.ix_(mask_g, mask_r)] * self.B[np.ix_(mask_g, mask_r)]
            
            loss4 = - torch.trace(self.softmax(self.C_g[mask_g,:].detach()).t() @ Aint @ self.softmax(self.C_r[mask_r,:].detach())) / torch.norm(self.softmax(self.C_g[mask_g,:].detach())) / torch.norm(Aint @ self.softmax(self.C_r[mask_r,:].detach()))
            
            loss5 = Aint.pow(2).mean()/self.A.pow(2).sum()
            
 
          
        elif mode == "A":
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g @  self.softmax(self.C_g[mask_g,:].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r @ self.softmax(self.C_r[mask_r, :].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = - torch.trace(self.A_r.t() @ self.A_g)/torch.norm(self.A_r)/torch.norm(self.A_g)
            loss4 = 0
            loss5 = 0
       
        elif mode == "b":
            with torch.no_grad():
                self.b_g[:, mask_g] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.A_g @ self.softmax(self.C_g)[mask_g,:].t() - self.b_1[mask_1,:], dim = 0)[None,:]/mask_1.shape[0]
                self.b_r[:, mask_r] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ self.A_r @ self.softmax(self.C_r[mask_r, :]).t() - self.b_2[mask_2,:], dim = 0)[None,:]/mask_2.shape[0]
                self.b_1[mask_1, :] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.A_g @  self.softmax(self.C_g)[mask_g,:].t() - self.b_g[:, mask_g], dim = 1)[:,None]/mask_g.shape[0]
                self.b_2[mask_2, :] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ self.A_r @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r], dim = 1)[:,None]/mask_r.shape[0]
            
                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0
                loss5 = 0
                
        elif mode == 'valid':
            with torch.no_grad():
                loss1 = (self.G - self.softmax(self.C_1) @ self.A_g @ self.softmax(self.C_g).t() - self.b_g - self.b_1).pow(2).mean()
                     
                loss2 = (self.R - self.softmax(self.C_2) @ self.A_r @ self.softmax(self.C_r).t() - self.b_r - self.b_2).pow(2).mean()
                 
                loss3 = - torch.trace(self.A_r.t() @ self.A_g)/torch.norm(self.A_r)/torch.norm(self.A_g)
                Aint = self.A * self.B
                loss4 = - torch.trace(self.softmax(self.C_g).t() @ Aint @ self.softmax(self.C_r)) /torch.norm(self.softmax(self.C_g)) / torch.norm(Aint @ self.softmax(self.C_r)) 
                loss5 = Aint.pow(2).sum()/self.A.pow(2).sum()
                     
        else:
            raise NotImplementedError
        loss = self.alpha[0] * loss1 + self.alpha[1] * loss2 + self.alpha[2] * loss3 + self.alpha[3] * loss4 + self.alpha[4] * loss5
        
        return loss, self.alpha[0] * loss1, self.alpha[1] * loss2, self.alpha[2] * loss3, self.alpha[3] * loss4, self.alpha[4] * loss5   
    
    def train_func(self, T):
        best_loss = 1e12
        count = 0
        
        for t in range(T):
            
            for mode in ['b']* 1 + ['C_12']*1 + ['C_gr']*1 + ['A']*1 + ['B']*1:
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
                    'loss 5: {:.5f}'.format(loss5.item())
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

                            
                            
                            
                            
######################################################################################################################
class cfrm_imp(Module):
    def __init__(self, counts, N=3, batch_size=1.0, interval=10, lr=1e-3, alpha = [1000, 1000, 100, 0.0, 0.0], binarize = False):
        super().__init__()
        self.N = N
        self.batch_size = batch_size
        self.interval = interval
        self.alpha = torch.FloatTensor(alpha).to(device)

        # data
        counts_rna = counts["rna"][0]
        counts_atac = counts["atac"][0]
        gact = counts["gact"][0]
        
        self.binarize = binarize
        if self.binarize:
            k = np.int(0.9 * counts_rna.shape[1])
            kth_index = np.argpartition(counts_rna, kth = k - 1, axis = 1)[:,(k-1)]
            kth_dist = np.take_along_axis(counts_rna, kth_index[:,None], axis = 1)
            counts_rna = ((counts_rna - kth_dist) >= 0).astype(np.float)
        else:
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
        Ainit = torch.randn((N,N))
        self.A_r = Parameter(Ainit)        
        self.A_g = Parameter(Ainit)

        self.C_1 = Parameter(torch.rand(self.G.shape[0], N))
        self.C_2 = Parameter(torch.rand(self.R.shape[0], N))
        # C_g is the average of C_r
        self.C_r = Parameter(torch.rand(self.R.shape[1], N))
        
        self.b_g = torch.zeros(1, self.G.shape[1]).to(device)
        # self.b_r = torch.zeros(1, self.R.shape[1]).to(device) 
        self.b_r = Parameter(torch.zeros(1, self.R.shape[1]))
        
        self.b_1 = torch.zeros(self.G.shape[0], 1).to(device)
        # self.b_2 = torch.zeros(self.R.shape[0], 1).to(device) 
        self.b_2 = Parameter(torch.zeros(self.R.shape[0],1))
        
        self.optimizer = opt.Adam(self.parameters(), lr=lr)
    
    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim = 1)
    
    @staticmethod
    def entropy_loss(C):
        loss = - (F.softmax(C, dim=1) - 0.5).pow(2).mean()
        
        return loss
    
    @staticmethod
    def cross_entropy(X_hat: Tensor, X: Tensor):
        assert torch.sum(X_hat <= 0) == 0
        assert torch.sum(X_hat >= 1) == 0
        return - (X * torch.log(X_hat) + (1 - X) * torch.log(1 - X_hat)).mean()
    
    def batch_loss(self, mode='C_c'):
        if mode != 'valid':
            mask_1 = np.random.choice(self.G.shape[0], int(self.G.shape[0] * self.batch_size), replace=False)
            mask_2 = np.random.choice(self.R.shape[0], int(self.R.shape[0] * self.batch_size), replace=False)
            mask_g = np.random.choice(self.G.shape[1], int(self.G.shape[1] * self.batch_size), replace=False)
            mask_r = np.random.choice(self.R.shape[1], int(self.R.shape[1] * self.batch_size), replace=False)
            
            
        if mode == 'C_12':
            if self.binarize:
                prob = torch.sigmoid(self.softmax(self.C_1[mask_1,:]) @ self.A_g.detach() @ (self.A @ self.softmax(self.C_r.detach()))[mask_g, :].t() + self.b_g[:, mask_g] + self.b_1[mask_1,:])
                loss1 = self.cross_entropy(prob, self.G[np.ix_(mask_1, mask_g)])
                
            else:
                loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.A_g.detach() @ (self.A @ self.softmax(self.C_r.detach()))[mask_g, :].t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

            prob = torch.sigmoid(self.softmax(self.C_2[mask_2,:]) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r,:].detach()).t() + self.b_r[:, mask_r] + self.b_2[mask_2,:])    
            loss2 = self.cross_entropy(prob, self.R[np.ix_(mask_2, mask_r)])
            
            loss3 = 0
            loss4 = utils.maximum_mean_discrepancy(self.softmax(self.C_1[mask_1,:]), self.softmax(self.C_2[mask_2,:]))
            
                
        elif mode == 'C_gr':
            if self.binarize:
                prob = torch.sigmoid(self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g.detach() @ (self.A @ self.softmax(self.C_r))[mask_g,:].t() + self.b_g[:, mask_g] + self.b_1[mask_1,:])
                loss1 = self.cross_entropy(prob, self.G[np.ix_(mask_1, mask_g)])
            
            else:
                loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g.detach() @ (self.A @ self.softmax(self.C_r))[mask_g,:].t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            prob = torch.sigmoid(self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r, :]).t() + self.b_r[:, mask_r] + self.b_2[mask_2,:])
            loss2 = self.cross_entropy(prob, self.R[np.ix_(mask_2, mask_r)])
            
            loss3 = 0 
            loss4 = 0 
            
        
        elif mode == "A":
            if self.binarize:
                prob = torch.sigmoid(self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g @ (self.A @ self.softmax(self.C_r.detach()))[mask_g,:].t() + self.b_g[:, mask_g] + self.b_1[mask_1,:])
                loss1 = self.cross_entropy(prob, self.G[np.ix_(mask_1, mask_g)])
            
            else:    
                loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g @ (self.A @ self.softmax(self.C_r.detach()))[mask_g,:].t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            prob = torch.sigmoid(self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r @ self.softmax(self.C_r[mask_r, :].detach()).t() + self.b_r[:, mask_r] + self.b_2[mask_2,:])
            loss2 = self.cross_entropy(prob, self.R[np.ix_(mask_2, mask_r)])
            
            loss3 = - torch.trace(self.A_r.t() @ self.A_g)/torch.norm(self.A_r)/torch.norm(self.A_g)
            loss4 = 0
       
        elif mode == "b":
            with torch.no_grad():
                if self.binarize:
                    pass
                else:
                    self.b_g[:, mask_g] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.A_g @ (self.A @ self.softmax(self.C_r))[mask_g,:].t() - self.b_1[mask_1,:], dim = 0)[None,:]/mask_1.shape[0]
                    self.b_1[mask_1, :] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.A_g @ (self.A @ self.softmax(self.C_r))[mask_g,:].t() - self.b_g[:, mask_g], dim = 1)[:,None]/mask_g.shape[0]

            
                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0
                
        elif mode == 'valid':
            with torch.no_grad():
                if self.binarize:
                    prob = torch.sigmoid(self.softmax(self.C_1) @ self.A_g @ (self.A @ self.softmax(self.C_r)).t() + self.b_g + self.b_1)
                    loss1 = self.cross_entropy(prob, self.G[np.ix_(mask_1, mask_g)])
                else:
                    loss1 = (self.G - self.softmax(self.C_1) @ self.A_g @ (self.A @ self.softmax(self.C_r)).t() - self.b_g - self.b_1).pow(2).mean()
                
                prob = torch.sigmoid(self.softmax(self.C_2) @ self.A_r @ self.softmax(self.C_r).t() + self.b_r + self.b_2)    
                loss2 = self.cross_entropy(prob, self.R)
                 
                loss3 = - torch.trace(self.A_r.t() @ self.A_g)/torch.norm(self.A_r)/torch.norm(self.A_g)
                loss4 = utils.maximum_mean_discrepancy(self.softmax(self.C_1), self.softmax(self.C_2))
            
                     
        else:
            raise NotImplementedError
        loss = self.alpha[0] * loss1 + self.alpha[1] * loss2 + self.alpha[2] * loss3 + self.alpha[3] * loss4
        
        return loss, self.alpha[0] * loss1, self.alpha[1] * loss2, self.alpha[2] * loss3, self.alpha[3] * loss4
    
    def train_func(self, T):
        best_loss = 1e12
        count = 0
        
        for t in range(T):
            losses1 = []
            losses2 = []
            losses3 = []
            
            for mode in ['C_12']*1 + ['C_gr']*1 + ['A']*1 + ['b']:
                loss, loss1, loss2, loss3, _ = self.batch_loss(mode)
                
                if (mode == 'C_12'):
                    losses1.append(loss1)
                if (mode == 'C_gr'):
                    losses2.append(loss2)
                if (mode == 'A'):
                    losses3.append(loss3)
                
                if mode != 'b':
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    

            if (t+1) % self.interval == 0:
                loss, loss1, loss2, loss3, loss4 = self.batch_loss('valid')
                
                print('Epoch {}, Validating Loss: {:.4f}'.format(t + 1, loss.item()))
                info = [
                    'loss 1: {:.5f}'.format(loss1.item()),
                    'loss 2: {:.5f}'.format(loss2.item()),
                    'loss 3: {:.5f}'.format(loss3.item()),
                    'loss 4: {:.5f}'.format(loss4.item())
                ]
                for i in info:
                    print("\t", i)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(self.state_dict(), f'../check_points/real_{self.N}.pt')
                    count = 0
                else:
                    count += 1
                    if count % 40 == 0:
                        self.optimizer.param_groups[0]['lr'] *= 0.5
                        print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < 1e-6:
                            break
                        else:
                            self.load_state_dict(torch.load(f'../check_points/real_{self.N}.pt'))
                            count = 0


                            