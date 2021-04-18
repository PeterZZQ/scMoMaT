import sys, os
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as opt
import pandas as pd
from torch.nn import Module, Parameter
from torch import softmax, log_softmax, Tensor

from sklearn.cluster import KMeans
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import time
import utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class cfrm_test(Module):
    def __init__(self, dir = "../data/simulated/2b3c_ziqi1/", N=3, batch_size=1.0, interval=10, lr=1e-3, alpha = [1, 1, 0.01, 1, 1e-2]):
        super().__init__()
        self.N = N
        self.batch_size = batch_size
        self.interval = interval
        self.alpha = torch.FloatTensor(alpha).to(device)

        # data
        counts_rna = pd.read_csv(dir + "GxC1.txt", sep = "\t", header = None).values.T
        counts_atac = pd.read_csv(dir + "RxC2.txt", sep = "\t", header = None).values.T
        gact = pd.read_csv(dir + "region2gene.txt", sep = "\t", header = None).values.T

        counts_rna = utils.preprocess(counts_rna, mode = "quantile", modality = "RNA")
        counts_atac = utils.preprocess(counts_atac, mode = "quantile", modality = "ATAC")
#         counts_atac = counts_atac/np.sum(counts_atac, axis = 1)[:,None]
#         counts_rna = counts_rna/np.sum(counts_rna, axis = 1)[:,None]
        gact = utils.preprocess(gact, mode = "gact")

        # non-negative, don't use standard scaler
        counts_rna = counts_rna/np.max(counts_rna)
        counts_atac = counts_atac/np.max(counts_atac)

        self.G = torch.FloatTensor(counts_rna).to(device)
        self.R = torch.FloatTensor(counts_atac).to(device)
        self.A = torch.FloatTensor(gact).to(device)

        # self.Ar = torch.eye(N).to(device)
        # self.Ar = Parameter(torch.randn((N,N)))        
        # self.Ag = torch.eye(N).to(device) 
        self.Ag = Parameter(torch.randn((N,N)))
        self.Ar = self.Ag

        self.C_1 = Parameter(torch.rand(self.G.shape[0], N))
        self.C_2 = Parameter(torch.rand(self.R.shape[0], N))
        self.C_g = Parameter(torch.rand(self.G.shape[1], N))
        self.C_r = Parameter(torch.rand(self.R.shape[1], N))
        
#         self.b_g = torch.zeros(1, self.G.shape[1]).to(device)
#         self.b_r = torch.zeros(1, self.R.shape[1]).to(device)
#         self.b_1 = torch.zeros(self.G.shape[0],1).to(device)
#         self.b_2 = torch.zeros(self.R.shape[0],1).to(device)

        self.b_g = Parameter(torch.rand(1, self.G.shape[1]))
        self.b_r = Parameter(torch.rand(1, self.R.shape[1]))
        self.b_1 = Parameter(torch.rand(self.G.shape[0],1))
        self.b_2 = Parameter(torch.rand(self.R.shape[0],1))
        
        
        # low rank decomp
        self.Pl = Parameter(torch.randn(self.G.shape[0], N))
        self.Pr = Parameter(torch.randn(N, self.R.shape[0]))
        
        self.meta_rna = pd.read_csv(os.path.join(dir, "cell_label1.txt"), sep = "\t")
        self.meta_atac = pd.read_csv(os.path.join(dir, "cell_label2.txt"), sep = "\t")

        self.optimizer = opt.Adam(self.parameters(), lr=lr)
    
    @staticmethod
    def softmax(X: Tensor):
        # return torch.relu(X)
        return torch.softmax(X, dim = 1)
    
    @staticmethod
    def act(X: Tensor):
        # not necessarily positive, so gene possibly negatively regulate the cell
        # return torch.relu(X)
        return X
    @staticmethod
    def entropy_loss(C):
#         loss = - F.softmax(C, dim=1) * F.log_softmax(C, dim=1)
#         loss = loss.sum(dim=1).mean()
        # loss = torch.sum(C, dim = 1).pow(2).sum()
        
        # New
        loss = - (F.softmax(C, dim=1) - 0.5).pow(2).mean()
        return loss
    
    @staticmethod
    def cross_entropy(C, C_hat):
        loss = - C * torch.log(C_hat/torch.max(C_hat)) - (1 - C) * torch.log(1 - C_hat/(torch.max(C_hat) + 1e-6))
        return torch.sum(loss)

    def batch_loss(self, mode='C_c'):
        if mode != 'valid':
            mask_1 = np.random.choice(self.G.shape[0], int(self.G.shape[0] * self.batch_size), replace=False)
            mask_2 = np.random.choice(self.R.shape[0], int(self.R.shape[0] * self.batch_size), replace=False)
            mask_g = np.random.choice(self.G.shape[1], int(self.G.shape[1] * self.batch_size), replace=False)
            mask_r = np.random.choice(self.R.shape[1], int(self.R.shape[1] * self.batch_size), replace=False)
            
        if mode == 'C_12':
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - (self.softmax(self.Pl[mask_1,:] @ self.Pr[:,mask_2]) @ self.softmax(self.C_2[mask_2,:]) @ self.act(self.Ag.detach()) @ self.softmax(self.C_g[mask_g, :].detach()).t()) - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - (self.softmax(self.C_2[mask_2,:]) @ self.act(self.Ar.detach()) @ self.softmax(self.C_r[mask_r,:].detach()).t()) - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()    
            
            loss3 = 0
            loss4 = 0 
            loss5 = 0 * self.entropy_loss(self.C_1[mask_1,:]) + 0 * self.entropy_loss(self.C_2[mask_2,:]) + 0 * self.entropy_loss(self.C_g[mask_g,:].detach()) + 0 * self.entropy_loss(self.C_r[mask_r,:].detach()) + self.entropy_loss(self.softmax(self.Pl[mask_1,:] @ self.Pr[:,mask_2]))
            
            # + self.b_1[mask_1,:].pow(2).mean() + self.b_2[mask_2,:].pow(2).mean() + self.b_g[:,mask_g].pow(2).mean() + self.b_r[:,mask_r].pow(2).mean()
                
        elif mode == 'C_gr':
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - (self.softmax(self.Pl[mask_1,:].detach() @ self.Pr[:,mask_2].detach()) @ self.softmax(self.C_2[mask_2,:].detach()) @ self.act(self.Ag.detach()) @ self.softmax(self.C_g[mask_g, :]).t()) - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
    
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - (self.softmax(self.C_2[mask_2,:].detach()) @ self.act(self.Ar.detach()) @ self.softmax(self.C_r[mask_r, :]).t()) - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = - torch.trace((self.softmax(self.C_g[mask_g,:]) / torch.norm(self.softmax(self.C_g[mask_g,:]))).t() @ self.A[np.ix_(mask_g, mask_r)] @ (self.softmax(self.C_r[mask_r,:]) / torch.norm(self.softmax(self.C_r[mask_r,:]))))
#             loss3 = (self.A[np.ix_(mask_g, mask_r)].t() @ self.softmax(self.C_g[mask_g,:]) - self.softmax(self.C_r[mask_r,:])).pow(2).mean()
            
            loss4 = 0 
            loss5 = 0 * self.entropy_loss(self.C_1[mask_1,:].detach()) + 0 * self.entropy_loss(self.C_2[mask_2,:].detach()) + 0 * self.entropy_loss(self.C_g[mask_g,:]) + self.entropy_loss(self.C_r[mask_r,:]) + self.entropy_loss(self.softmax(self.Pl[mask_1,:].detach() @ self.Pr[:,mask_2].detach())) # + self.b_1[mask_1,:].pow(2).mean() + self.b_2[mask_2,:].pow(2).mean() + self.b_g[:,mask_g].pow(2).mean() + self.b_r[:,mask_r].pow(2).mean() 
            
        
        elif mode == "A":
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - (self.softmax(self.Pl[mask_1,:].detach() @ self.Pr[:,mask_2].detach()) @ self.softmax(self.C_2[mask_2,:].detach()) @ self.act(self.Ag) @ self.softmax(self.C_g[mask_g, :].detach()).t()) - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - (self.softmax(self.C_2[mask_2,:].detach()) @ self.act(self.Ar) @ self.softmax(self.C_r[mask_r, :].detach()).t()) - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = 0
            # loss4 = -torch.trace((self.Ar/torch.norm(self.Ar)).t() @ (self.Ag)/torch.norm(self.Ag)) 
            loss4 = (self.Ar - self.Ag).pow(2).mean()
            loss5 = 0
       
        elif mode == 'b':
            with torch.no_grad():
                self.b_g[:, mask_g] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.Pl[mask_1,:] @ self.Pr[:,mask_2]) @ self.softmax(self.C_2[mask_2,:]) @ self.Ag @ self.softmax(self.C_g[mask_g, :]).t() - self.b_1[mask_1,:], dim = 0)[None,:]/mask_1.shape[0]
                self.b_r[:, mask_r] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ self.Ar @ self.softmax(self.C_r[mask_r, :]).t() - self.b_2[mask_2,:], dim = 0)[None,:]/mask_2.shape[0]
                self.b_1[mask_1, :] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.Pl[mask_1,:] @ self.Pr[:,mask_2]) @ self.softmax(self.C_2[mask_2,:]) @ self.Ag @ self.softmax(self.C_g[mask_g, :]).t() - self.b_g[:, mask_g], dim = 1)[:,None]/mask_g.shape[0]
                self.b_2[mask_2, :] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ self.Ar @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r], dim = 1)[:,None]/mask_r.shape[0]
            
                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0       
                loss5 = 0
   
        elif mode == 'valid':
            with torch.no_grad():
                loss1 = (self.G - (self.softmax(self.Pl @ self.Pr) @ self.softmax(self.C_2) @ self.act(self.Ag) @ self.softmax(self.C_g).t()) - self.b_g - self.b_1).pow(2).mean()
                     
                loss2 = (self.R - (self.softmax(self.C_2) @ self.act(self.Ar) @ self.softmax(self.C_r).t()) - self.b_r - self.b_2).pow(2).mean()
                
                loss3 = - torch.trace((self.softmax(self.C_g) / torch.norm(self.softmax(self.C_g))).t() @ self.A @ (self.softmax(self.C_r) / torch.norm(self.softmax(self.C_r))))
#                 loss3 = (self.A.t() @ self.softmax(self.C_g) - self.softmax(self.C_r)).pow(2).mean()
                
                # loss4 = - torch.trace((self.Ar/torch.norm(self.Ar)).t() @ (self.Ag)/torch.norm(self.Ag))
                loss4 = (self.Ar - self.Ag).pow(2).mean()
                loss5 = 0 * self.entropy_loss(self.C_1) + 0 * self.entropy_loss(self.C_2) + 0 * self.entropy_loss(self.C_g) + self.entropy_loss(self.C_r) + self.entropy_loss(self.softmax(self.Pl @ self.Pr)) 
                # + self.b_1.pow(2).mean() + self.b_2.pow(2).mean() + self.b_g.pow(2).mean() + self.b_r.pow(2).mean() 
                     
        else:
            raise NotImplementedError
        loss = self.alpha[0] * loss1 + self.alpha[1] * loss2 + self.alpha[2] * loss3 + self.alpha[3] * loss4 + self.alpha[4] * loss5
               
        return loss, self.alpha[0] * loss1, self.alpha[1] * loss2, self.alpha[2] * loss3, self.alpha[3] * loss4, self.alpha[4] * loss5
    #
    def train_func(self, T):
        best_loss = 1e12
        count = 0
        for t in range(T):
            self.optimizer.zero_grad()
            for mode in ['C_12', 'C_gr', 'A']:
                loss, *_ = self.batch_loss(mode)
                if mode != 'b':
                    loss.backward()
                    self.optimizer.step()
               
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
                    torch.save(self.state_dict(), f'../check_points/real_{self.N}.pt')
                    count = 0
                else:
                    count += 1
                    if count % 20 == 0:
                        self.optimizer.param_groups[0]['lr'] *= 0.5
                        print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < 1e-4:
                            break
                        else:
                            self.load_state_dict(torch.load(f'../check_points/real_{self.N}.pt'))
                            count = 0
        
        
############################################################################################################################### 
class cfrm_best(Module):
    def __init__(self, dir = "../data/simulated/2b3c_ziqi1/", N=3, batch_size=1.0, interval=10, lr=1e-3, alpha = [1000, 1000, 100, 0.0]):
        super().__init__()
        self.N = N
        self.batch_size = batch_size
        self.interval = interval
        self.alpha = torch.FloatTensor(alpha).to(device)

        # data
        counts_rna = pd.read_csv(dir + "GxC1.txt", sep = "\t", header = None).values.T
        counts_atac = pd.read_csv(dir + "RxC2.txt", sep = "\t", header = None).values.T
        gact = pd.read_csv(dir + "region2gene.txt", sep = "\t", header = None).values.T

        counts_rna = utils.preprocess(counts_rna, mode = "quantile", modality = "RNA")
        counts_atac = utils.preprocess(counts_atac, mode = "quantile", modality = "ATAC")
        gact = utils.preprocess(gact, mode = "gact")

        # non-negative, don't use standard scaler
        counts_rna = counts_rna/np.max(counts_rna)
        counts_atac = counts_atac/np.max(counts_atac)

        self.G = torch.FloatTensor(counts_rna).to(device)
        self.R = torch.FloatTensor(counts_atac).to(device)
        self.A = torch.FloatTensor(gact).to(device)

        self.Ai = Parameter(torch.randn((N,N)))        

        self.C_1 = Parameter(torch.rand(self.G.shape[0], N))
        self.C_2 = Parameter(torch.rand(self.R.shape[0], N))
        self.C_g = Parameter(torch.rand(self.G.shape[1], N))
        self.C_r = Parameter(torch.rand(self.R.shape[1], N))
        
#         self.b_g = Parameter(torch.rand(1, self.G.shape[1]))
#         self.b_r = Parameter(torch.rand(1, self.R.shape[1]))
#         self.b_1 = Parameter(torch.rand(self.G.shape[0],1))
#         self.b_2 = Parameter(torch.rand(self.R.shape[0],1))
        
        self.b_g = torch.zeros(1, self.G.shape[1]).to(device)
        self.b_r = torch.zeros(1, self.R.shape[1]).to(device)
        self.b_1 = torch.zeros(self.G.shape[0],1).to(device)
        self.b_2 = torch.zeros(self.R.shape[0],1).to(device)
        
        self.meta_rna = pd.read_csv(os.path.join(dir, "cell_label1.txt"), sep = "\t")
        self.meta_atac = pd.read_csv(os.path.join(dir, "cell_label2.txt"), sep = "\t")

        self.optimizer = opt.Adam(self.parameters(), lr=lr)
    
    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim = 1)
        # return torch.relu(X)
    
    @staticmethod
    def entropy_loss(C):
        # loss = - F.softmax(C, dim=1) * F.log_softmax(C, dim=1)
        # loss = loss.sum(dim=1).mean()
        
        # loss = torch.relu(C).sum(dim = 1).pow(2).mean()
        
        # New
        loss = - (F.softmax(C, dim=1) - 0.5).pow(2).mean()
        
        return loss
    
    def batch_loss(self, mode='C_c'):
        if mode != 'valid':
            mask_1 = np.random.choice(self.G.shape[0], int(self.G.shape[0] * self.batch_size), replace=False)
            mask_2 = np.random.choice(self.R.shape[0], int(self.R.shape[0] * self.batch_size), replace=False)
            mask_g = np.random.choice(self.G.shape[1], int(self.G.shape[1] * self.batch_size), replace=False)
            mask_r = np.random.choice(self.R.shape[1], int(self.R.shape[1] * self.batch_size), replace=False)
            
        if mode == 'C_12':
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - (self.softmax(self.C_1[mask_1,:]) @ self.Ai.detach() @ self.softmax(self.C_g[mask_g, :].detach()).t()) - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - (self.softmax(self.C_2[mask_2,:]) @ self.Ai.detach() @ self.softmax(self.C_r[mask_r,:].detach()).t()) - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()    
            
            loss3 = 0

            loss4 = self.entropy_loss(self.C_1[mask_1,:]) + self.entropy_loss(self.C_2[mask_2,:]) + self.entropy_loss(self.C_g[mask_g,:].detach()) + self.entropy_loss(self.C_r[mask_r,:].detach())
            
                
        elif mode == 'C_gr':
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - (self.softmax(self.C_1[mask_1,:].detach()) @ self.Ai.detach() @ self.softmax(self.C_g[mask_g, :]).t()) - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
    
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - (self.softmax(self.C_2[mask_2,:].detach()) @ self.Ai.detach() @ self.softmax(self.C_r[mask_r, :]).t()) - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = - torch.trace((self.softmax(self.C_g[mask_g,:]) / torch.norm(self.softmax(self.C_g[mask_g,:]))).t() @ self.A[np.ix_(mask_g, mask_r)] @ (self.softmax(self.C_r[mask_r,:]) / torch.norm(self.softmax(self.C_r[mask_r,:]))))
            
            loss4 = self.entropy_loss(self.C_1[mask_1,:].detach()) + self.entropy_loss(self.C_2[mask_2,:].detach()) + self.entropy_loss(self.C_g[mask_g,:]) + self.entropy_loss(self.C_r[mask_r,:])
            
        
        elif mode == "A":
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - (self.softmax(self.C_1[mask_1,:].detach()) @ self.Ai @ self.softmax(self.C_g[mask_g, :].detach()).t()) - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - (self.softmax(self.C_2[mask_2,:].detach()) @ self.Ai @ self.softmax(self.C_r[mask_r, :].detach()).t()) - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = 0
            
            loss4 = 0
       
        elif mode == "b":
            with torch.no_grad():
                self.b_g[:, mask_g] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.Ai @ self.softmax(self.C_g[mask_g, :]).t() - self.b_1[mask_1,:], dim = 0)[None,:]/mask_1.shape[0]
                self.b_r[:, mask_r] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ self.Ai @ self.softmax(self.C_r[mask_r, :]).t() - self.b_2[mask_2,:], dim = 0)[None,:]/mask_2.shape[0]
                self.b_1[mask_1, :] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.Ai @ self.softmax(self.C_g[mask_g, :]).t() - self.b_g[:, mask_g], dim = 1)[:,None]/mask_g.shape[0]
                self.b_2[mask_2, :] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ self.Ai @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r], dim = 1)[:,None]/mask_r.shape[0]
            
                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0
                
        elif mode == 'valid':
            with torch.no_grad():
                
#                 self.b_g = torch.sum(self.G - self.softmax(self.C_1) @ self.Ai @ self.softmax(self.C_g).t() - self.b_1, dim = 0)[None,:]/self.G.shape[0]
#                 self.b_r = torch.sum(self.R - self.softmax(self.C_2) @ self.Ai @ self.softmax(self.C_r).t() - self.b_2, dim = 0)[None,:]/self.R.shape[0]
                
#                 self.b_1 = torch.sum(self.G - self.softmax(self.C_1) @ self.Ai @ self.softmax(self.C_g).t() - self.b_g, dim = 1)[:,None]/self.G.shape[1]
#                 self.b_2 = torch.sum(self.R - self.softmax(self.C_2) @ self.Ai @ self.softmax(self.C_r).t() - self.b_r, dim = 1)[:,None]/self.R.shape[1]
                
                loss1 = (self.G - (self.softmax(self.C_1) @ self.Ai @ self.softmax(self.C_g).t()) - self.b_g - self.b_1).pow(2).mean()
                     
                loss2 = (self.R - (self.softmax(self.C_2) @ self.Ai @ self.softmax(self.C_r).t()) - self.b_r - self.b_2).pow(2).mean()
                
                loss3 = - torch.trace((self.softmax(self.C_g) / torch.norm(self.softmax(self.C_g))).t() @ self.A @ (self.softmax(self.C_r) / torch.norm(self.softmax(self.C_r))))
                
                loss4 = self.entropy_loss(self.C_1) + self.entropy_loss(self.C_2) + self.entropy_loss(self.C_g) + self.entropy_loss(self.C_r)
                     
        else:
            raise NotImplementedError
        loss = self.alpha[0] * loss1 + self.alpha[1] * loss2 + self.alpha[2] * loss3 + self.alpha[3] * loss4
               
        return loss, self.alpha[0] * loss1, self.alpha[1] * loss2, self.alpha[2] * loss3, self.alpha[3] * loss4
    
    def train_func(self, T):
        best_loss = 1e12
        count = 0
        for t in range(T):
            self.optimizer.zero_grad()
            for mode in ['C_12', 'C_gr', 'A', 'b']:
                loss, *_ = self.batch_loss(mode)
                if mode != 'b':
                    loss.backward()
                    self.optimizer.step()
                    
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
                    if count % 20 == 0:
                        self.optimizer.param_groups[0]['lr'] *= 0.5
                        print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < 1e-4:
                            break
                        else:
                            self.load_state_dict(torch.load(f'../check_points/real_{self.N}.pt'))
                            count = 0


                            
    
class classicCFRM(Module):
    def __init__(self, dir, N=3, batch_size=100, lr=1e-3, dropout=0.1, init = "random"):
        super().__init__()
        self.batch_size = batch_size
        self.alpha = torch.FloatTensor([1, 1, 1])
        self.dropout = dropout
        self.N = N
        # data
        counts_rna = np.loadtxt(os.path.join(dir, 'GxC1.txt')).T
        counts_atac = np.loadtxt(os.path.join(dir, 'RxC2.txt')).T
        
        counts_rna = utils.preprocess(counts = counts_rna, mode = "quantile", modality = "RNA")
        counts_atac = utils.preprocess(counts = counts_atac, mode = "quantile", modality= "ATAC")

        self.G = torch.FloatTensor(counts_rna)
        self.R = torch.FloatTensor(counts_atac)
        self.A = torch.FloatTensor(np.loadtxt(os.path.join(dir, 'RxG.txt')))
        assert self.A.shape[0] == self.R.shape[1]
        assert self.A.shape[1] == self.G.shape[1]
        self.label_g = torch.LongTensor(np.loadtxt(os.path.join(dir, 'gene_label.txt')))
        self.label_c1 = torch.LongTensor(np.loadtxt(os.path.join(dir, 'cell_label_C1.txt'), skiprows=1, usecols=[1]))
        self.label_c2 = torch.LongTensor(np.loadtxt(os.path.join(dir, 'cell_label_C2.txt'), skiprows=1, usecols=[1]))

        # learnable parameters
        if init == "svd":
            u_g, s_g, v_g = torch.svd(self.G)
            self.C_1 = u_g[:, :N]
            self.C_g = v_g[:, :N]        

            u_r, s_r, v_r = torch.svd(self.R)
            self.C_2 = u_r[:, :N]
            self.C_r = v_r[:, :N]

        if init == "random":
            self.C_1, _ = torch.qr(torch.randn((self.G.shape[0],N)))
            self.C_2, _ = torch.qr(torch.randn((self.R.shape[0],N)))
            self.C_g, _ = torch.qr(torch.randn((self.G.shape[1],N)))
            self.C_r, _ = torch.qr(torch.randn((self.R.shape[1],N)))


        self.A_1g = self.C_1.t() @ self.G @ self.C_g
        self.A_2r = self.C_2.t() @ self.R @ self.C_r
        self.A_rg = self.C_r.t() @ self.A @ self.C_g

        loss = self.loss()
        for l in loss:
            print(l.item())
        
       
    def loss(self):
        loss1 = (self.G - self.C_1 @ self.A_1g @ self.C_g.t()).pow(2).mean()
        loss2 = (self.R - self.C_2 @ self.A_2r @ self.C_r.t()).pow(2).mean()
        loss3 = (self.A - self.C_r @ self.A_rg @ self.C_g.t()).pow(2).mean()
        loss = self.alpha[0] * loss1 + self.alpha[1] * loss2 + self.alpha[2] * loss3

        return loss, self.alpha[0] * loss1, self.alpha[1] * loss2, self.alpha[2] * loss3

    def train_func(self, T):
        for t in range(T):
            for mode in ['C_1', 'C_2', 'C_r', 'C_g']:
                if mode == 'C_1':
                    M = self.alpha[0] * (self.G @ self.C_g @ self.C_g.t() @ self.G.t())
                    s_g, u_g = torch.eig(M, eigenvectors=True)
                    self.C_1 = u_g[:, :self.N]

                elif mode == 'C_2':
                    M = self.alpha[1] * (self.R @ self.C_r @ self.C_r.t() @ self.R.t())
                    s_g, u_g = torch.eig(M, eigenvectors=True)
                    self.C_2 = u_g[:, :self.N]

                elif mode == 'C_r':
                    M = self.alpha[1] * (self.R.t() @ self.C_2 @ self.C_2.t() @ self.R) + self.alpha[2] * (self.A @ self.C_g @ self.C_g.t() @ self.A.t())
                    s_g, u_g = torch.eig(M, eigenvectors=True)
                    self.C_r = u_g[:, :self.N]

                elif mode == 'C_g':
                    M = self.alpha[0] * (self.G.t() @ self.C_1 @ self.C_1.t() @ self.G) + self.alpha[2] * (self.A.t() @ self.C_r @ self.C_r.t() @ self.A)
                    s_g, u_g = torch.eig(M, eigenvectors=True)
                    self.C_g = u_g[:, :self.N]
            
            self.A_1g = self.C_1.t() @ self.G @ self.C_g
            self.A_2r = self.C_2.t() @ self.R @ self.C_r
            self.A_rg = self.C_r.t() @ self.A @ self.C_g
            
            loss, loss1, loss2, loss3 = self.loss()
            print('Epoch {}, Training Loss: {:.4f}'.format(t+1, loss.item()))
            info = [
                'loss RNA: {:.5f}'.format(loss1.item()),
                'loss ATAC: {:.5f}'.format(loss2.item()),
                'loss gene act: {:.5f}'.format(loss3.item())
            ]
            for i in info:
                print("\t", i)




                
"""
class cfrmModel(Module):
    def __init__(self, dir, N=3, batch_size=100, lr=1e-3, dropout=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.alpha = torch.FloatTensor([1, 1, 1, 1, 1, 1])
        self.dropout = dropout
        # data
        counts_rna = np.loadtxt(os.path.join(dir, 'GxC1.txt')).T
        counts_atac = np.loadtxt(os.path.join(dir, 'RxC2.txt')).T

        counts_rna = utils.preprocess(counts=counts_rna, mode="quantile", modality="RNA")
        counts_atac = utils.preprocess(counts=counts_atac, mode="quantile", modality="ATAC")

        self.G = torch.FloatTensor(counts_rna)
        self.R = torch.FloatTensor(counts_atac)
        self.A = torch.FloatTensor(np.loadtxt(os.path.join(dir, 'RxG.txt')))
        assert self.A.shape[0] == self.R.shape[1]
        assert self.A.shape[1] == self.G.shape[1]
        self.label_g = torch.LongTensor(np.loadtxt(os.path.join(dir, 'gene_label.txt')))
        self.label_c1 = torch.LongTensor(np.loadtxt(os.path.join(dir, 'cell_label_C1.txt'), skiprows=1, usecols=[1]))
        self.label_c2 = torch.LongTensor(np.loadtxt(os.path.join(dir, 'cell_label_C2.txt'), skiprows=1, usecols=[1]))

        # learnable parameters
        self.D_gr = Parameter(torch.ones(N, 1))
        u_g, s_g, v_g = torch.svd(self.G)
        self.C_1 = Parameter(u_g[:, :N])
        self.C_g = Parameter(v_g[:, :N])
        self.A_1g = self.C_1.t() @ self.G @ self.C_g
        u_r, s_r, v_r = torch.svd(self.R)
        self.C_2 = Parameter(u_r[:, :N])
        self.C_r = Parameter(v_r[:, :N])
        self.A_2r = self.C_2.t() @ self.R @ self.C_r
        self.D_12 = torch.diag(s_g[:N] / s_r[:N])
        self.W = Parameter(torch.eye(N))

        self.optimizer = opt.Adam(self.parameters(), lr=lr)

    @staticmethod
    def orthogonal_loss(A):
        return (A.t() @ A - torch.eye(A.shape[1])).pow(2).sum()

    def entropy_loss(self, C):
        z = C @ self.W
        loss = - F.softmax(z, dim=1) * F.log_softmax(z, dim=1)
        return loss.sum()

    def batch_loss(self, mode='C_c'):
        if mode == 'C_c':
            loss1 = (self.G - self.C_1 @ self.A_1g.detach() @ self.C_g.detach().t()).pow(2).mean()
            loss2 = (self.R - self.C_2 @ self.A_2r.detach() @ self.C_r.detach().t()).pow(2).mean()
            loss5 = sum(map(self.orthogonal_loss, [self.C_1, self.C_2]))
            loss6 = sum(map(self.entropy_loss, [self.C_1, self.C_2]))
            loss3, loss4 = 0, 0
        elif mode == 'C_r':
            loss2 = (self.R - self.C_2.detach() @ self.A_2r.detach() @ self.C_r.t()).pow(2).mean()
            loss3 = (self.A - self.C_r @ (self.D_gr.detach() * self.C_g.detach().t())).pow(2).mean()
            loss5 = sum(map(self.orthogonal_loss, [self.C_r]))
            loss6 = 0
            loss1, loss4 = 0, 0
        elif mode == 'C_g':
            loss1 = (self.G - self.C_1.detach() @ self.A_1g.detach() @ self.C_g.t()).pow(2).mean()
            loss3 = (self.A - self.C_r.detach() @ (self.D_gr.detach() * self.C_g.t())).pow(2).mean()
            loss5 = sum(map(self.orthogonal_loss, [self.C_g]))
            loss6 = 0
            loss2, loss4 = 0, 0
        elif mode == 'valid':
            loss1 = (self.G - self.C_1 @ self.A_1g @ self.C_g.t()).pow(2).mean().detach()
            loss2 = (self.R - self.C_2 @ self.A_2r @ self.C_r.t()).pow(2).mean().detach()
            loss3 = (self.A - self.C_r @ (self.D_gr * self.C_g.t())).pow(2).mean().detach()
            loss4 = (self.A_1g - self.D_12 * self.A_2r).pow(2).mean().detach()
            loss5 = sum(map(self.orthogonal_loss, [self.C_1, self.C_2, self.C_g, self.C_r])).detach()
            loss6 = sum(map(self.entropy_loss, [self.C_1, self.C_2])).detach()
        else:
            raise NotImplementedError
        loss = self.alpha[0] * loss1 + self.alpha[1] * loss2 + self.alpha[2] * loss3 + self.alpha[3] * loss4 + \
               self.alpha[4] * loss5 + self.alpha[5] * loss6

        return loss, self.alpha[0] * loss1, self.alpha[1] * loss2, self.alpha[2] * loss3, self.alpha[3] * loss4, \
               self.alpha[4] * loss5, self.alpha[5] * loss6

    def train_func(self, T):
        best_loss = 1e12
        count = 0
        for t in range(T):
            self.optimizer.zero_grad()
            for mode in ['C_c']:
            # for mode in ['C_c', 'C_r', 'C_g']:
                loss, *_ = self.batch_loss(mode)
                loss.backward()
                self.optimizer.step()
            self.A_1g = self.C_1.t() @ self.G @ self.C_g
            self.A_2r = self.C_2.t() @ self.R @ self.C_r
            self.D_12 = torch.diag((self.A_1g * self.A_2r).sum(dim=1) / self.A_2r.pow(2).sum(dim=1)).detach()
            loss, loss1, loss2, loss3, loss4, loss5, loss6 = self.batch_loss('valid')
            print('Epoch {}, Training Loss: {:.4f}'.format(t + 1, loss.item()))
            info = [
                'loss RNA: {:.5f}'.format(loss1.item()),
                'loss ATAC: {:.5f}'.format(loss2.item()),
                'loss gene act: {:.5f}'.format(loss3.item()),
                'loss merge: {:.5f}'.format(loss4.item()),
                'loss ortho: {:.5f}'.format(loss5.item()),
                'loss entropy: {:.5f}'.format(loss6.item())
            ]
            for i in info:
                print("\t", i)
            if loss.item() < best_loss:
                best_loss = loss.item()
                count = 0
            else:
                count += 1
                if count % 20 == 0:
                    self.optimizer.param_groups[0]['lr'] *= 0.5
                    print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                    if self.optimizer.param_groups[0]['lr'] < 1e-4:
                        break
                    else:
                        count = 0



class cfrmSparseModel(Module):
    def __init__(self, dir, N=3, batch_size=1.0, interval=10, lr=1e-3, init='svd'):
        super().__init__()
        self.batch_size = batch_size
        self.interval = interval
        self.alpha = torch.FloatTensor([1, 1, 1, 1, 1, 1])
        # data
        counts_rna = sp.load_npz(os.path.join(dir, 'C1xG.npz'))
        counts_atac = sp.load_npz(os.path.join(dir, 'C2xR.npz')).astype(np.float32)
        Cor = sp.load_npz(os.path.join(dir, 'GxR.npz')).astype(np.float32)

        self.G = torch.FloatTensor(counts_rna.todense())
        self.R = torch.FloatTensor(counts_atac.todense())
        self.M = torch.FloatTensor(Cor.todense())

        if init == 'svd':
            self.A = Parameter(torch.eye(N))
            u_g, s_g, v_g = torch.svd_lowrank(self.G, N)
            self.C_1 = Parameter(u_g)
            self.A_1g_l = Parameter(s_g.sqrt().reshape(-1, 1))
            self.A_1g_r = Parameter(s_g.sqrt().reshape(1, -1))
            self.C_g = Parameter(v_g)
            u_r, s_r, v_r = torch.svd_lowrank(self.R, N)
            self.C_2 = Parameter(u_r)
            self.A_2r_l = Parameter(s_r.sqrt().reshape(-1, 1))
            self.A_2r_r = Parameter(s_r.sqrt().reshape(1, -1))
            self.C_r = Parameter(v_r)
            self.A_gr_l = Parameter(torch.ones(N, 1))
            self.A_gr_r = Parameter(torch.ones(1, N))
        else:
            self.C_1 = Parameter(torch.randn(self.G.shape[0], N))
            self.C_2 = Parameter(torch.randn(self.R.shape[0], N))
            self.C_g = Parameter(torch.randn(self.G.shape[1], N))
            self.C_r = Parameter(torch.randn(self.R.shape[1], N))
            self.A = Parameter(torch.randn(N, N))
            self.A_1g_l = Parameter(torch.randn(N, 1))
            self.A_1g_r = Parameter(torch.randn(1, N))
            self.A_2r_l = Parameter(torch.randn(N, 1))
            self.A_2r_r = Parameter(torch.randn(1, N))
            self.A_gr_l = Parameter(torch.randn(N, 1))
            self.A_gr_r = Parameter(torch.randn(1, N))
        assert self.M.shape[0] == self.G.shape[1]
        assert self.M.shape[1] == self.R.shape[1]
        self.meta_rna = pd.read_csv(os.path.join(dir, "meta_rna.csv"), index_col=0)
        self.meta_atac = pd.read_csv(os.path.join(dir, "meta_atac.csv"), index_col=0)
        self.regions = pd.read_csv(os.path.join(dir, "regions.txt"), header=None)
        self.genes = pd.read_csv(os.path.join(dir, "genes.txt"), header=None)

        self.optimizer = opt.Adam(self.parameters(), lr=lr)

        with torch.no_grad():
            loss, *_ = self.batch_loss('valid')
            print('Initial Loss is {:.5f}'.format(loss.item()))


    def batch_loss(self, mode='C_c'):
        mask_1 = np.random.choice(self.G.shape[0], int(self.G.shape[0] * self.batch_size), replace=False)
        mask_2 = np.random.choice(self.R.shape[0], int(self.R.shape[0] * self.batch_size), replace=False)
        mask_g = np.random.choice(self.G.shape[1], int(self.G.shape[1] * self.batch_size), replace=False)
        mask_r = np.random.choice(self.R.shape[1], int(self.R.shape[1] * self.batch_size), replace=False)
        if mode == 'C_12':
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.C_1[mask_1, :] @ (self.A_1g_l * self.A * self.A_1g_r).detach() @
                     self.C_g[mask_g].detach().t()).pow(2).mean()
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.C_2[mask_2] @ (self.A_2r_l * self.A * self.A_2r_r).detach() @
                     self.C_r[mask_r].detach().t()).pow(2).mean()
            loss4 = (self.C_1[mask_1].mean(dim=0) - self.C_2[mask_2].mean(dim=0)).square().sum()
            loss3 = 0
        elif mode == 'C_gr':
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.C_1[mask_1].detach() @ (self.A_1g_l * self.A * self.A_1g_r).detach() @ self.C_g[mask_g].t()).pow(2).mean()
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.C_2[mask_2].detach() @ (self.A_2r_l * self.A * self.A_2r_r).detach() @ self.C_r[mask_r].t()).pow(2).mean()
            loss3 = (self.M[np.ix_(mask_g, mask_r)] - self.C_g[mask_g] @ (self.A_gr_l * self.A * self.A_gr_r).detach() @ self.C_r[mask_r].t()).pow(2).mean()
            loss4 = 0
        elif mode == 'A':
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.C_1[mask_1].detach() @ (self.A_1g_l * self.A * self.A_1g_r) @ self.C_g[mask_g].detach().t()).pow(2).mean()
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.C_2[mask_2].detach() @ (self.A_2r_l * self.A * self.A_2r_r) @ self.C_r[mask_r].detach().t()).pow(2).mean()
            loss3 = (self.M[np.ix_(mask_g, mask_r)] - self.C_g[mask_g].detach() @ (self.A_gr_l * self.A * self.A_gr_r) @ self.C_r[mask_r].detach().t()).pow(2).mean()
            loss4 = 0
        elif mode == 'valid':
            with torch.no_grad():
                loss1 = (self.G - self.C_1 @ (self.A_1g_l * self.A * self.A_1g_r) @ self.C_g.t()).pow(2).mean()
                loss2 = (self.R - self.C_2 @ (self.A_2r_l * self.A * self.A_2r_r) @ self.C_r.t()).pow(2).mean()
                loss3 = (self.M - self.C_g @ (self.A_gr_l * self.A * self.A_gr_r) @ self.C_r.t()).pow(2).mean()
                loss4 = (self.C_1.mean(dim=0) - self.C_2.mean(dim=0)).abs().sum()
        else:
            raise NotImplementedError
        loss = self.alpha[0] * loss1 + self.alpha[1] * loss2 + self.alpha[2] * loss3 + self.alpha[3] * loss4

        return loss, self.alpha[0] * loss1, self.alpha[1] * loss2, self.alpha[2] * loss3, self.alpha[3] * loss4
    #
    def train_func(self, T):
        best_loss = 1e12
        count = 0
        for t in range(T):
            self.optimizer.zero_grad()
            # for mode in ['C_c']:
            for mode in ['C_12', 'C_gr', 'C_12', 'C_gr', 'A']:
                loss, *_ = self.batch_loss(mode)
                loss.backward()
                self.optimizer.step()
            if (t+1) % self.interval == 0:
                loss, loss1, loss2, loss3, loss4 = self.batch_loss('valid')
                print('Epoch {}, Training Loss: {:.4f}'.format(t + 1, loss.item()))
                info = [
                    'loss RNA: {:.5f}'.format(loss1.item()),
                    'loss ATAC: {:.5f}'.format(loss2.item()),
                    'loss gene act: {:.5f}'.format(loss3.item()),
                    'loss merge: {:.5f}'.format(loss4.item()),
                ]
                for i in info:
                    print("\t", i)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(self.state_dict(), '../check_points/real_cfrm.pt')
                    count = 0
                else:
                    count += 1
                    if count % 20 == 0:
                        self.optimizer.param_groups[0]['lr'] *= 0.5
                        print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < 1e-4:
                            break
                        else:
                            self.load_state_dict(torch.load('../check_points/real_cfrm.pt'))
                            count = 0

class sparse_nmf_sgd(Module):
    def __init__(self, dir = "../data/simulated/2b3c_ziqi1/", N=3, batch_size=1.0, interval=10, lr=1e-3, alpha = [1, 1, 0.01, 1, 1e-2]):
        super().__init__()
        self.N = N
        self.batch_size = batch_size
        self.interval = interval
        self.alpha = torch.FloatTensor(alpha).to(device)

        # data
        counts_rna = pd.read_csv(dir + "GxC1.txt", sep = "\t", header = None).values.T
        counts_atac = pd.read_csv(dir + "RxC2.txt", sep = "\t", header = None).values.T
        gact = pd.read_csv(dir + "region2gene.txt", sep = "\t", header = None).values.T

        counts_rna = utils.preprocess(counts_rna, mode = "quantile", modality = "RNA")
        counts_atac = utils.preprocess(counts_atac, mode = "quantile", modality = "ATAC")
        gact = utils.preprocess(gact, mode = "gact")

        # non-negative, don't use standard scaler
        counts_rna = counts_rna/np.max(counts_rna)
        counts_atac = counts_atac/np.max(counts_atac)

        self.G = torch.FloatTensor(counts_rna).to(device)
        self.R = torch.FloatTensor(counts_atac).to(device)
        self.A = torch.FloatTensor(gact).to(device)

        self.Ar = Parameter(torch.randn((N,N)))        
        self.Ag = Parameter(torch.randn((N,N)))

        self.C_1 = Parameter(torch.rand(self.G.shape[0], N))
        self.C_2 = Parameter(torch.rand(self.R.shape[0], N))
        self.C_g = Parameter(torch.rand(self.G.shape[1], N))
        self.C_r = Parameter(torch.rand(self.R.shape[1], N))
        
        self.b_g = torch.zeros(1, self.G.shape[1]).to(device)
        self.b_r = torch.zeros(1, self.R.shape[1]).to(device)
        # self.b_g = Parameter(torch.rand(1, self.G.shape[1]))
        # self.b_r = Parameter(torch.rand(1, self.R.shape[1]))

        self.b_1 = torch.zeros(self.G.shape[0],1).to(device)
        self.b_2 = torch.zeros(self.R.shape[0],1).to(device)
#         self.b_1 = Parameter(torch.rand(self.G.shape[0],1))
#         self.b_2 = Parameter(torch.rand(self.R.shape[0],1))
        
        self.meta_rna = pd.read_csv(os.path.join(dir, "cell_label1.txt"), sep = "\t")
        self.meta_atac = pd.read_csv(os.path.join(dir, "cell_label2.txt"), sep = "\t")

        self.optimizer = opt.Adam(self.parameters(), lr=lr)
    
    @staticmethod
    def softmax(X: Tensor):
        return torch.relu(X)
    
    @staticmethod
    def sparse(C):
        loss = torch.sum(C, dim = 1).pow(2).sum()
        return loss
    
    @staticmethod
    def cross_entropy(C, C_hat):
        loss = - C * torch.log(C_hat/torch.max(C_hat)) - (1 - C) * torch.log(1 - C_hat/(torch.max(C_hat) + 1e-6))
        return torch.sum(loss)

    def batch_loss(self, mode='C_c'):
        if mode != 'valid':
            mask_1 = np.random.choice(self.G.shape[0], int(self.G.shape[0] * self.batch_size), replace=False)
            mask_2 = np.random.choice(self.R.shape[0], int(self.R.shape[0] * self.batch_size), replace=False)
            mask_g = np.random.choice(self.G.shape[1], int(self.G.shape[1] * self.batch_size), replace=False)
            mask_r = np.random.choice(self.R.shape[1], int(self.R.shape[1] * self.batch_size), replace=False)
            
        if mode == 'C_12':
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - (self.softmax(self.C_1[mask_1,:]) @ self.Ag.detach() @ self.softmax(self.C_g[mask_g, :].detach()).t()) - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - (self.softmax(self.C_2[mask_2,:]) @ self.Ar.detach() @ self.softmax(self.C_r[mask_r,:].detach()).t()) - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()    
            
            loss3 = 0
            loss4 = 0 
            loss5 = self.entropy_loss(self.C_1[mask_1,:]) + self.entropy_loss(self.C_2[mask_2,:]) + self.entropy_loss(self.C_g[mask_g,:].detach()) + self.entropy_loss(self.C_r[mask_r,:].detach())
                
        elif mode == 'C_gr':
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - (self.softmax(self.C_1[mask_1,:].detach()) @ self.Ag.detach() @ self.softmax(self.C_g[mask_g, :]).t()) - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
    
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - (self.softmax(self.C_2[mask_2,:].detach()) @ self.Ar.detach() @ self.softmax(self.C_r[mask_r, :]).t()) - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = - torch.trace((self.softmax(self.C_g[mask_g,:]) / torch.norm(self.softmax(self.C_g[mask_g,:]))).t() @ self.A[np.ix_(mask_g, mask_r)] @ (self.softmax(self.C_r[mask_r,:]) / torch.norm(self.softmax(self.C_r[mask_r,:]))))
            
            loss4 = 0 
            loss5 = self.entropy_loss(self.C_1[mask_1,:].detach()) + self.entropy_loss(self.C_2[mask_2,:].detach()) + self.entropy_loss(self.C_g[mask_g,:]) + self.entropy_loss(self.C_r[mask_r,:])
            
        
        elif mode == "A":
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - (self.softmax(self.C_1[mask_1,:].detach()) @ self.Ag @ self.softmax(self.C_g[mask_g, :].detach()).t()) - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - (self.softmax(self.C_2[mask_2,:].detach()) @ self.Ar @ self.softmax(self.C_r[mask_r, :].detach()).t()) - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = 0
            loss4 = -torch.trace((self.Ar/torch.norm(self.Ar)).t() @ (self.Ag)/torch.norm(self.Ag)) #(self.Ar - self.Ag).pow(2).mean()
            loss5 = 0
            
   
        elif mode == 'valid':
            with torch.no_grad():
                loss1 = (self.G - (self.softmax(self.C_1) @ self.Ag @ self.softmax(self.C_g).t()) - self.b_g - self.b_1).pow(2).mean()
                     
                loss2 = (self.R - (self.softmax(self.C_2) @ self.Ar @ self.softmax(self.C_r).t()) - self.b_r - self.b_2).pow(2).mean()
                
                loss3 = - torch.trace((self.softmax(self.C_g) / torch.norm(self.softmax(self.C_g))).t() @ self.A @ (self.softmax(self.C_r) / torch.norm(self.softmax(self.C_r))))
                
                loss4 = - torch.trace((self.Ar/torch.norm(self.Ar)).t() @ (self.Ag)/torch.norm(self.Ag))#(self.Ar - self.Ag).pow(2).mean()
                loss5 = self.entropy_loss(self.C_1) + self.entropy_loss(self.C_2) + self.entropy_loss(self.C_g) + self.entropy_loss(self.C_r)
                     
        else:
            raise NotImplementedError
        loss = self.alpha[0] * loss1 + self.alpha[1] * loss2 + self.alpha[2] * loss3 + self.alpha[3] * loss4 + self.alpha[4] * loss5
               
        return loss, self.alpha[0] * loss1, self.alpha[1] * loss2, self.alpha[2] * loss3, self.alpha[3] * loss4, self.alpha[4] * loss5
    #
    def train_func(self, T):
        best_loss = 1e12
        count = 0
        for t in range(T):
            self.optimizer.zero_grad()
            for mode in ['C_12', 'C_gr']:
                loss, *_ = self.batch_loss(mode)
                loss.backward()
                self.optimizer.step()
                    
                
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
                    torch.save(self.state_dict(), f'../check_points/real_{self.N}.pt')
                    count = 0
                else:
                    count += 1
                    if count % 20 == 0:
                        self.optimizer.param_groups[0]['lr'] *= 0.5
                        print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < 1e-4:
                            break
                        else:
                            self.load_state_dict(torch.load(f'../check_points/real_{self.N}.pt'))
                            count = 0


class sc_mf(Module):
    def __init__(self, dir, N=3, batch_size=1.0, interval=10, lr=1e-3, init='svd'):
        super().__init__()
        self.N = N
        self.batch_size = batch_size
        self.interval = interval
        self.alpha = torch.FloatTensor([1000, 1000, 1000, 1, 0]).to(device)
        # data
        counts_rna = pd.read_csv(dir + "GxC1.txt", sep = "\t", header = None).values.T
        counts_atac = pd.read_csv(dir + "RxC2.txt", sep = "\t", header = None).values.T
        gact = pd.read_csv(dir + "region2gene.txt", sep = "\t", header = None).values.T
     
        counts_rna = utils.preprocess(counts_rna, mode = "quantile", modality = "RNA")
        counts_atac = utils.preprocess(counts_atac, mode = "quantile", modality = "ATAC")
        gact = utils.preprocess(gact, mode = "gact")
        
        counts_rna = counts_rna/np.max(counts_rna)
        counts_atac = counts_atac/np.max(counts_atac)

        self.G = torch.FloatTensor(counts_rna).to(device)
        self.R = torch.FloatTensor(counts_atac).to(device)
        self.A = torch.FloatTensor(gact).to(device)

        self.s_g = Parameter(torch.ones(1, self.G.shape[1]))
        self.s_r = Parameter(torch.ones(1, self.R.shape[1]))
        self.b_g = Parameter(torch.zeros(1, self.G.shape[1]))
        self.b_r = Parameter(torch.zeros(1, self.R.shape[1]))
        self.C_1 = Parameter(torch.randn(self.G.shape[0], N))
        self.C_2 = Parameter(torch.randn(self.R.shape[0], N))
        self.C_g = Parameter(torch.randn(self.G.shape[1], N))
        self.C_r = Parameter(torch.randn(self.R.shape[1], N))

        self.meta_rna = pd.read_csv(os.path.join(dir, "cell_label1.txt"), sep = "\t")
        self.meta_atac = pd.read_csv(os.path.join(dir, "cell_label2.txt"), sep = "\t")

        self.optimizer = opt.Adam(self.parameters(), lr=lr)


    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim=1)

    @staticmethod
    def entropy_loss(C):
        loss = - F.softmax(C, dim=1) * F.log_softmax(C, dim=1)
        return loss.sum(dim=1).mean()

    def batch_loss(self, mode='C_c'):
        if mode != 'valid':
            mask_1 = np.random.choice(self.G.shape[0], int(self.G.shape[0] * self.batch_size), replace=False)
            mask_2 = np.random.choice(self.R.shape[0], int(self.R.shape[0] * self.batch_size), replace=False)
            mask_g = np.random.choice(self.G.shape[1], int(self.G.shape[1] * self.batch_size), replace=False)
            mask_r = np.random.choice(self.R.shape[1], int(self.R.shape[1] * self.batch_size), replace=False)
            
        if mode == 'C_12':
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.s_g[:, mask_g] * (self.softmax(self.C_1[mask_1]) @
                     self.softmax(self.C_g[mask_g].detach()).t()) - self.b_g[:, mask_g]).pow(2).mean()
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.s_r[:, mask_r] * (self.softmax(self.C_2[mask_2]) @
                     self.softmax(self.C_r[mask_r].detach()).t()) - self.b_r[:, mask_r]).pow(2).mean()
            loss3 = 0
            loss4 = self.entropy_loss(self.C_1[mask_1]) + self.entropy_loss(self.C_2[mask_2])
            loss5 = 0
        elif mode == 'C_gr':
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.s_g[:, mask_g] * (self.softmax(self.C_1[mask_1].detach()) @
                     self.softmax(self.C_g[mask_g]).t()) - self.b_g[:, mask_g]).pow(2).mean()
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.s_r[:, mask_r] * (self.softmax(self.C_2[mask_2].detach()) @
                     self.softmax(self.C_r[mask_r]).t()) - self.b_r[:, mask_r]).pow(2).mean()
            loss3 = (self.A[:, mask_r].t() @ self.softmax(self.C_g) - self.softmax(self.C_r[mask_r])).pow(2).mean()
            # loss3 = (self.A[mask_g] @ self.C_r - self.C_g[mask_g]).pow(2).mean()
            loss4 = 0
            loss5 = self.entropy_loss(self.C_g[mask_g]) + self.entropy_loss(self.C_r[mask_g])
        elif mode == 'valid':
            with torch.no_grad():
                loss1 = (self.G - self.s_g * (self.softmax(self.C_1) @
                         self.softmax(self.C_g).t()) - self.b_g).pow(2).mean()
                loss2 = (self.R - self.s_r * (self.softmax(self.C_2.detach()) @
                         self.softmax(self.C_r).t()) - self.b_r).pow(2).mean()
                loss3 = (self.A.t() @ self.softmax(self.C_g) - self.softmax(self.C_r)).pow(2).mean()
                loss4 = self.entropy_loss(self.C_1) + self.entropy_loss(self.C_2)
                loss5 = self.entropy_loss(self.C_g) + self.entropy_loss(self.C_r)
        else:
            raise NotImplementedError
        loss = self.alpha[0] * loss1 + self.alpha[1] * loss2 + self.alpha[2] * loss3 + self.alpha[3] * loss4 + self.alpha[4] * loss5

        return loss, self.alpha[0] * loss1, self.alpha[1] * loss2, self.alpha[2] * loss3, self.alpha[3] * loss4, self.alpha[4] * loss5
    #
    def train_func(self, T):
        best_loss = 1e12
        count = 0
        for t in range(T):
            self.optimizer.zero_grad()
            ################### separate ##############################
#             loss = 0
#             for it in range(500):
#                 self.optimizer.zero_grad()
#                 mode = 'C_12'
#                 loss_pre = loss
#                 loss, *_ = self.batch_loss(mode)
#                 loss.backward()
#                 self.optimizer.step()
#                 if torch.abs(loss_pre - loss) < 1e-3:
#                     print(it)
#                     break
            
#             loss = 0
#             for it in range(500):
#                 mode = 'C_gr'
#                 self.optimizer.zero_grad()
#                 loss_pre = loss
#                 loss, *_ = self.batch_loss(mode)
#                 loss.backward()
#                 self.optimizer.step()
#                 if torch.abs(loss_pre - loss) < 1e-3:
#                     print(it)
#                     break            
            ################### joint #################################
            
            for mode in ['C_12', 'C_gr']:
                loss, *_ = self.batch_loss(mode)
                loss.backward()
                self.optimizer.step()

            ###########################################################
            if (t+1) % self.interval == 0:
                loss, loss1, loss2, loss3, loss4, loss5 = self.batch_loss('valid')
                print('Epoch {}, Validating Loss: {:.4f}'.format(t + 1, loss.item()))
                info = [
                    'loss RNA: {:.5f}'.format(loss1.item()),
                    'loss ATAC: {:.5f}'.format(loss2.item()),
                    'loss gene act: {:.5f}'.format(loss3.item()),
                    'loss sparse: {:.5f}'.format(loss4.item() + loss5.item()),
                ]
                for i in info:
                    print("\t", i)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(self.state_dict(), f'../check_points/real_{self.N}.pt')
                    count = 0
                else:
                    count += 1
                    if count % 20 == 0:
                        self.optimizer.param_groups[0]['lr'] *= 0.5
                        print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < 1e-4:
                            break
                        else:
                            self.load_state_dict(torch.load(f'../check_points/real_{self.N}.pt'))
                            count = 0


class sparse_nmf(Module):
    def __init__(self, dir = "../data/simulated/2b3c_ziqi1/", N=3, batch_size=1.0, interval=10, lr=1e-3, alpha = [1, 1, 0.01, 1, 1e-2]):
        super().__init__()
        self.N = N
        self.batch_size = batch_size
        self.interval = interval
        self.alpha = torch.FloatTensor(alpha).to(device)
        # data
        counts_rna = pd.read_csv(dir + "GxC1.txt", sep = "\t", header = None).values.T
        counts_atac = pd.read_csv(dir + "RxC2.txt", sep = "\t", header = None).values.T
        gact = pd.read_csv(dir + "region2gene.txt", sep = "\t", header = None).values

        counts_rna = utils.preprocess(counts_rna, mode = "quantile", modality = "RNA")
        counts_atac = utils.preprocess(counts_atac, mode = "quantile", modality = "ATAC")
        # gact = utils.preprocess(gact, mode = "gact")

        self.G = torch.FloatTensor(counts_rna).to(device)
        self.R = torch.FloatTensor(counts_atac).to(device)
        self.A = torch.FloatTensor(gact).to(device)
        self.A = self.A/(torch.norm(self.A, dim = 0)[None, :] + 1e-6)
        # print(torch.norm(self.A, dim = 0))

        self.Ar = Parameter(torch.randn((N,N)))
        self.Ag = Parameter(torch.randn((N,N)))
        
        self.C_1 = torch.randn(self.R.shape[0], N).to(device)
        self.C_2 = torch.randn(self.G.shape[0], N).to(device)
        self.C_r = torch.randn(self.R.shape[1], N).to(device)
        self.C_g = torch.randn(self.G.shape[1], N).to(device)
        
        self.C_1 = (self.C_1 >= 0) * self.C_1
        self.C_2 = (self.C_2 >= 0) * self.C_2
        self.C_r = (self.C_r >= 0) * self.C_r
        self.C_g = (self.C_g >= 0) * self.C_g

        self.meta_rna = pd.read_csv(os.path.join(dir, "cell_label1.txt"), sep = "\t")
        self.meta_atac = pd.read_csv(os.path.join(dir, "cell_label2.txt"), sep = "\t")
        
        self.delta = 0

        self.optimizer = opt.Adam(self.parameters(), lr=lr)

    def batch_loss(self, mode='A'):

        if mode == "A":
            loss1 = (self.R - self.C_1.detach() @ self.Ar @ self.C_r.detach().t()).pow(2).mean()
            
            loss2 = (self.G - self.C_2.detach() @ self.Ag @ self.C_g.detach().t()).pow(2).mean()
            
            loss3 = 0
            loss4 = (self.Ar - self.Ag).pow(2).mean()
            loss5 = 0
           

        elif mode == 'valid':
            with torch.no_grad():
                assert torch.sum(self.C_1  < 0) == 0
                assert torch.sum(self.C_2  < 0) == 0
                assert torch.sum(self.C_g  < 0) == 0
                assert torch.sum(self.C_r  < 0) == 0
                
                loss1 = (self.R - self.C_1 @ self.Ar @ self.C_r.t()).pow(2).mean()
                
                loss2 = (self.G - self.C_2 @ self.Ag @ self.C_g.t()).pow(2).mean()
                
                loss3 = (self.A @ self.C_g - self.C_r).pow(2).mean() # - torch.trace(self.C_r.t() @ self.A @ self.C_g)
                
                loss4 = (self.Ar - self.Ag).pow(2).mean()
                
                loss5 = self.C_1.sum(dim = 1).pow(2).sum() + self.C_2.sum(dim = 1).pow(2).sum() # + self.C_r.sum(dim = 1).pow(2).sum() + self.C_g.sum(dim = 1).pow(2).sum() 
                # loss5 = torch.sum(torch.abs(self.C_1)) + torch.sum(torch.abs(self.C_2))
        else:
            raise NotImplementedError
        loss = self.alpha[0] * loss1 + self.alpha[1] * loss2 + self.alpha[2] * loss3 + self.alpha[3] * loss4 + self.alpha[4] * loss5
               
        return loss, loss1, loss2, loss3, loss4, loss5
    

    def train_func(self, T):
        best_loss = 1e12
        count = 0
        for t in range(T):
            with torch.no_grad():
                ############### C1 ###################
                M = torch.cat((self.C_r @ self.Ar.t(), torch.sqrt(self.alpha[4]) * torch.ones(1, self.N).to(device)), dim = 0)
                Ra = torch.cat((self.R.t(), torch.zeros(1, self.R.shape[0]).to(device)), dim = 0)
                self.C_1 = Ra.t() @ M @ torch.inverse(M.t() @ M + self.delta * torch.eye(M.shape[1]).cuda())
                self.C_1 = (self.C_1 >= 0) * self.C_1
                
                loss, loss1, loss2, loss3, loss4, loss5 = self.batch_loss('valid')
                print('Epoch {}, Validating Loss: {:.4f}'.format(t + 1, loss.item()))
                info = [
                    'loss RNA: {:.5f}'.format(loss1.item()),
                    'loss ATAC: {:.5f}'.format(loss2.item()),
                    'loss gene act: {:.5f}'.format(loss3.item()),
                    'loss interact: {:.5f}'.format(loss4.item()),
                    'loss sparse: {:.5f}'.format(loss5.item()),
                ]
                for i in info:
                    print("\t", i)

                ############### C2 ###################
                M = torch.cat((self.C_g @ self.Ag.t(), torch.sqrt(self.alpha[4]) * torch.ones(1, self.N).to(device)), dim = 0)
                Ga = torch.cat((self.G.t(), torch.zeros(1, self.G.shape[0]).to(device)), dim = 0)
                self.C_2 = Ga.t() @ M @ torch.inverse(M.t() @ M + self.delta * torch.eye(M.shape[1]).cuda())
                self.C_2 = (self.C_2 >= 0) * self.C_2

                loss, loss1, loss2, loss3, loss4, loss5 = self.batch_loss('valid')
                print('Epoch {}, Validating Loss: {:.4f}'.format(t + 1, loss.item()))
                info = [
                    'loss RNA: {:.5f}'.format(loss1.item()),
                    'loss ATAC: {:.5f}'.format(loss2.item()),
                    'loss gene act: {:.5f}'.format(loss3.item()),
                    'loss interact: {:.5f}'.format(loss4.item()),
                    'loss sparse: {:.5f}'.format(loss5.item()),
                ]
                for i in info:
                    print("\t", i)
                ############### Cr ###################
                M = torch.cat((self.C_1 @ self.Ar, 0 * torch.sqrt(self.alpha[4]) * torch.ones(1, self.N).to(device), torch.eye(self.N).to(device)), dim = 0)
                Ra = torch.cat((self.R, torch.zeros(1, self.R.shape[1]).to(device), self.C_g.t() @ self.A.t()))
                self.C_r = Ra.t() @ M @ torch.inverse(M.t() @ M + self.delta * torch.eye(M.shape[1]).cuda())
                
                # self.C_r = (torch.inverse(2 * self.Ar.t() @ self.C_1.t() @ self.C_1 @ self.Ar + 2 * torch.sqrt(self.alpha[4]) * torch.ones(self.N, self.N).to(device)) @ (2 * self.Ar.t() @ self.C_1.t() @ self.R + self.alpha[2] * self.C_g.t() @ self.A.t())).t()
                self.C_r = (self.C_r >= 0) * self.C_r

                ############### Cg ###################
                M = torch.cat((self.C_2 @ self.Ag, 0 * torch.sqrt(self.alpha[4]) * torch.ones(1, self.N).to(device), torch.eye(self.N).to(device)), dim = 0)
                Ra = torch.cat((self.G, torch.zeros(1, self.G.shape[1]).to(device), self.C_r.t() @ self.A))
                self.C_g = Ra.t() @ M @ torch.inverse(M.t() @ M + self.delta * torch.eye(M.shape[1]).cuda())
                
                # self.C_g = (torch.inverse(2 * self.Ag.t() @ self.C_2.t() @ self.C_2 @ self.Ag + 2 * torch.sqrt(self.alpha[4]) * torch.ones(self.N, self.N).to(device)) @ (2 * self.Ag.t() @ self.C_2.t() @ self.G + self.alpha[2] * self.C_r.t() @ self.A)).t()
                self.C_g = (self.C_g >= 0) * self.C_g
            
            loss = 0
            for it in range(100):
                self.optimizer.zero_grad()
                loss_pre = loss
                loss, *_ = self.batch_loss(mode = "A")
                loss.backward()
                self.optimizer.step()
                if torch.abs(loss_pre - loss) < 1e-3:
                    print(it)
                    break   
              
            ###########################################################
            if (t+1) % self.interval == 0:
                loss, loss1, loss2, loss3, loss4, loss5 = self.batch_loss('valid')
                print('Epoch {}, Validating Loss: {:.4f}'.format(t + 1, loss.item()))
                info = [
                    'loss RNA: {:.5f}'.format(loss1.item()),
                    'loss ATAC: {:.5f}'.format(loss2.item()),
                    'loss gene act: {:.5f}'.format(loss3.item()),
                    'loss interact: {:.5f}'.format(loss4.item()),
                    'loss sparse: {:.5f}'.format(loss5.item()),
                ]
                for i in info:
                    print("\t", i)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(self.state_dict(), f'../check_points/real_{self.N}.pt')
                    count = 0
                else:
                    count += 1
                    if count % 20 == 0:
                        self.optimizer.param_groups[0]['lr'] *= 0.5
                        print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < 1e-4:
                            break
                        else:
                            self.load_state_dict(torch.load(f'../check_points/real_{self.N}.pt'))
                            count = 0
        


"""