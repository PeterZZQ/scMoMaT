import sys, os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import itertools
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as opt
from torch.nn import Module, Parameter
from torch import softmax, log_softmax, Tensor

import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sparse_tensor(X):
    X = sp.coo_matrix(X)
    X = torch.sparse_coo_tensor(indices = [[x for x in X.row], [x for x in X.col]], 
                                values = [x for x in X.data], size = X.shape)
    return X

def slicing_sparse(X, rows, cols):
    batch_X = torch.zeros((rows.shape[0], cols.shape[0]))
    for i, row in enumerate(rows):
        for j, col in enumerate(cols): 
            if (row in X.indices_()[0,:]) & (col in X.indices_()[1,:]):
                batch_X[i,j] = X[row, col]
    
    return batch_X

class cfrm_new(Module):
    def __init__(self, counts, N=3, batch_size=1.0, interval=10, lr=1e-3, alpha = [1000, 1000, 100, 100, 0.0], seed = None, init = None, sparse = False):
        super().__init__()
        self.N1 = N
        self.N2 = N
        self.batch_size = batch_size
        self.interval = interval
        self.alpha = torch.FloatTensor(alpha).to(device)
        self.sparse = sparse
        
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
        
        if self.sparse:
            A_sp = sparse_tensor(gact).to(device)
            R_sp = sparse_tensor(counts_atac).to(device)
            G_sp = sparse_tensor(counts_rna).to(device)
            
        else:
            self.G = torch.FloatTensor(counts_rna)
            self.R = torch.FloatTensor(counts_atac)
            self.A = torch.FloatTensor(gact)
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
        
        self.optimizer = opt.Adam(self.parameters(), lr = lr)

    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim = 1)
    
    @staticmethod
    def entropy_loss(C):
        loss = - (F.softmax(C, dim = 1) - 0.5).pow(2).mean()
        
        return loss
    
    def batch_loss(self, mode):
        if mode != 'valid':
            mask_1 = np.random.choice(self.G.shape[0], int(self.G.shape[0] * self.batch_size), replace=False)
            mask_2 = np.random.choice(self.R.shape[0], int(self.R.shape[0] * self.batch_size), replace=False)
            mask_g = np.random.choice(self.G.shape[1], int(self.G.shape[1] * self.batch_size), replace=False)
            mask_r = np.random.choice(self.R.shape[1], int(self.R.shape[1] * self.batch_size), replace=False)
            
            if self.sparse:
                mbatch_G = slicing_sparse(X).to(device)
                mbatch_R = slicing_sparse(R).to(device)
                mbatch_A = slicing_sparse(A).to(device)
                
            else:
                mbatch_G = self.G[np.ix_(mask_1, mask_g)].to(device)
                mbatch_R = self.R[np.ix_(mask_2, mask_r)].to(device)
                mbatch_A = self.A[np.ix_(mask_g, mask_r)].to(device)
                
            
            
        if mode == 'C_12':
            
            loss1 = (mbatch_G - self.softmax(self.C_1[mask_1,:]) @ self.A_g.detach() @ self.softmax(self.C_g[mask_g, :].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (mbatch_R - self.softmax(self.C_2[mask_2,:]) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r,:].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()    
            
            loss3 = 0
            loss4 = 0 
            loss5 = 0
            
                
        elif mode == 'C_gr':
            
            loss1 = (mbatch_G - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g.detach() @ self.softmax(self.C_g)[mask_g, :].t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

            loss2 = (mbatch_R - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = 0 
            
            Aint = mbatch_A * self.B[np.ix_(mask_g, mask_r)].detach()
            
            loss4 = - torch.trace(self.softmax(self.C_g[mask_g,:]).t() @ Aint @ self.softmax(self.C_r[mask_r,:])) /torch.norm(self.softmax(self.C_g[mask_g,:])) / torch.norm(Aint @ self.softmax(self.C_r[mask_r,:]))
            
            loss5 = 0

        elif mode == 'B':
            
            loss1 = (mbatch_G - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g.detach() @ self.softmax(self.C_g.detach())[mask_g, :].t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

            loss2 = (mbatch_R - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r, :].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = 0
            
            Aint = mbatch_A * self.B[np.ix_(mask_g, mask_r)]
            
            loss4 = - torch.trace(self.softmax(self.C_g[mask_g,:].detach()).t() @ Aint @ self.softmax(self.C_r[mask_r,:].detach())) / torch.norm(self.softmax(self.C_g[mask_g,:].detach())) / torch.norm(Aint @ self.softmax(self.C_r[mask_r,:].detach()))
            
            loss5 = Aint.abs().sum()/torch.sum(self.A)
            
 
          
        elif mode == "A":
            
            loss1 = (mbatch_G - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g @  self.softmax(self.C_g[mask_g,:].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (mbatch_R - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r @ self.softmax(self.C_r[mask_r, :].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = - torch.trace(self.A_r.t() @ self.A_g)/torch.norm(self.A_r)/torch.norm(self.A_g)
            loss4 = 0
            loss5 = 0
       
        elif mode == "b":
            with torch.no_grad():
                self.b_g[:, mask_g] = torch.sum(mbatch_G - self.softmax(self.C_1[mask_1,:]) @ self.A_g @ self.softmax(self.C_g)[mask_g,:].t() - self.b_1[mask_1,:], dim = 0)[None,:]/mask_1.shape[0]
                self.b_r[:, mask_r] = torch.sum(mbatch_R - self.softmax(self.C_2[mask_2,:]) @ self.A_r @ self.softmax(self.C_r[mask_r, :]).t() - self.b_2[mask_2,:], dim = 0)[None,:]/mask_2.shape[0]
                self.b_1[mask_1, :] = torch.sum(mbatch_G - self.softmax(self.C_1[mask_1,:]) @ self.A_g @ self.softmax(self.C_g)[mask_g,:].t() - self.b_g[:, mask_g], dim = 1)[:,None]/mask_g.shape[0]
                self.b_2[mask_2, :] = torch.sum(mbatch_R - self.softmax(self.C_2[mask_2,:]) @ self.A_r @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r], dim = 1)[:,None]/mask_r.shape[0]
            
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
