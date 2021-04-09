import sys, os
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as opt
import pandas as pd
from torch import nn
from torch.nn import Module, Parameter
from torch import softmax, log_softmax, Tensor

from sklearn.cluster import KMeans
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import time
import utils
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import Parameter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NCF(Module):
    def __init__(self, nfeatures_b1, nfeatures_b2, nfeatures_rna, nfeatures_atac, embed_features, A):
        super().__init__()
        self.proj_b1 = nn.Linear(nfeatures_b1, embed_features, bias = False)
        self.proj_b2 = nn.Linear(nfeatures_b2, embed_features, bias = False)
        self.proj_rna1 = nn.Linear(nfeatures_rna, 2 * embed_features, bias = False)
        self.proj_rna2 = nn.Linear(2 * embed_features, embed_features, bias = False)
        
#         self.proj_b1 = Parameter(torch.randn(nfeatures_b1, embed_features))
#         self.proj_b2 = Parameter(torch.randn(nfeatures_b2, embed_features))
#         self.proj_rna1 = Parameter(torch.randn(nfeatures_rna, embed_features))
#         self.proj_rna2 = Parameter(torch.randn(nfeatures_rna, embed_features))
        
        
        # initialize gene activity matrix
        self.gene_act = Parameter(A)
        
        self.w_prod1 = Parameter(torch.randn(embed_features, 1))
        self.w_add1 = Parameter(torch.randn(2 * embed_features, 1))
        # self.w_add1 = torch.zeros(2 * embed_features, 1).to(device)
        
        self.w_prod2 = Parameter(torch.randn(embed_features, 1))
        self.w_add2 = Parameter(torch.randn(2 * embed_features, 1))
        # self.w_add2 = torch.zeros(2 * embed_features, 1).to(device)
        
        self.bias_1 = Parameter(torch.randn(1,1))
        self.bias_2 = Parameter(torch.randn(1,1))
#         self.bias_1 = 0
#         self.bias_2 = 0

    @staticmethod
    def act(X: torch.Tensor, mode = "softmax"):
        if mode == "softmax":
            return torch.softmax(X, dim=1)
        elif mode == "sigmoid":
            return torch.sigmoid(X)
        else:
            return X
        # return torch.exp(X)
        
    def forward(self, u1, u2, v1, v2):
        embed_u1 = self.act(self.proj_b1(u1), mode = "linear")
        embed_v1 = self.act(self.proj_rna2(F.relu(self.proj_rna1(v1))), mode = "linear")
        
#         embed_u1 = u1 @ self.proj_b1
#         embed_v1 = v1 @ self.proj_rna1
        
        embed_u1_exp = embed_u1[:,None,:].expand(*(-1,embed_v1.shape[0],-1))
        embed_v1_exp = embed_v1[None,:,:].expand(*(embed_u1.shape[0],-1,-1))

        
        out1 = F.relu((embed_u1_exp * embed_v1_exp) @ self.w_prod1 + torch.cat((embed_u1_exp, embed_v1_exp), dim = -1) @ self.w_add1 + self.bias_1)

        embed_u2 = self.act(self.proj_b2(u2), mode = "linear")
        embed_v2 = self.act(self.proj_rna2(F.relu(self.proj_rna1(v2 @ self.gene_act))), mode = "linear")
#         embed_u2 = u2 @ self.proj_b2
#         embed_v2 = v2 @ self.gene_act @ self.proj_rna2
        
        embed_u2_exp = embed_u2[:,None,:].expand(*(-1,embed_v2.shape[0],-1))
        embed_v2_exp = embed_v2[None,:,:].expand(*(embed_u2.shape[0],-1,-1))
        
        out2 = torch.sigmoid((embed_u2_exp * embed_v2_exp) @ self.w_prod2 + torch.cat((embed_u2_exp, embed_v2_exp), dim = -1) @ self.w_add2 + self.bias_2)
        
        return out1.squeeze(), out2.squeeze(), embed_u1, embed_v1, embed_u2, embed_v2


class deep_cfrm(Module):
    def __init__(self, dir = "../data/simulated/2b3c_ziqi1/", N=3, batch_size=1.0, interval=10, lr=1e-3, alpha = torch.FloatTensor([1000, 1000, 0.01, 0])):
        super().__init__()
        self.N = N
        self.batch_size = batch_size
        self.interval = interval
        self.alpha = alpha.to(device)
        # data
        counts_rna = pd.read_csv(dir + "GxC1.txt", sep = "\t", header = None).values.T
        counts_atac = pd.read_csv(dir + "RxC2.txt", sep = "\t", header = None).values.T
        gact = pd.read_csv(dir + "region2gene.txt", sep = "\t", header = None).values

        counts_rna = utils.preprocess(counts_rna, mode = "quantile", modality = "RNA")
        counts_atac = utils.preprocess(counts_atac, mode = "quantile", modality = "ATAC")
        gact = utils.preprocess(gact, mode = "gact")

        # non-negative, don't use standard scaler
        counts_rna = counts_rna/np.max(counts_rna)
        counts_atac = counts_atac/np.max(counts_atac)

        self.G = torch.FloatTensor(counts_rna).to(device)
        self.R = torch.FloatTensor(counts_atac).to(device)
        self.A = torch.FloatTensor(gact).to(device)
       
        self.ncf = NCF(nfeatures_b1 = self.G.shape[0], nfeatures_b2 = self.R.shape[0], 
                       nfeatures_rna = self.G.shape[1], nfeatures_atac = self.R.shape[1],
                       embed_features = N, A = self.A)
        
        self.meta_rna = pd.read_csv(os.path.join(dir, "cell_label1.txt"), sep = "\t")
        self.meta_atac = pd.read_csv(os.path.join(dir, "cell_label2.txt"), sep = "\t")


        self.optimizer = opt.Adam(self.ncf.parameters(), lr = lr, weight_decay = 1e-2)
        
    @staticmethod
    def entropy_loss(C):
        loss = - C * torch.log(C)
        return loss.sum(dim=1).mean()
    
    @staticmethod
    def recon_loss(x_est, x, mode = "RNA"):
        if mode == "ATAC":
            loss = torch.sum(- x * torch.log(x_est) - (1 - x) * torch.log(1 - x_est))
        else:
            loss = (x_est - x).pow(2).mean()
        
        return loss
            

    def train_func(self, T):
        best_loss = 1e12
        count = 0
        
        for t in range(T):
            self.optimizer.zero_grad()
            
            mask_1 = torch.tensor(np.random.choice(self.G.shape[0], int(self.G.shape[0] * self.batch_size), replace = False)).to(device)
            mask_2 = torch.tensor(np.random.choice(self.R.shape[0], int(self.R.shape[0] * self.batch_size), replace = False)).to(device)
            mask_g = torch.tensor(np.random.choice(self.G.shape[1], int(self.G.shape[1] * self.batch_size), replace = False)).to(device)
            mask_r = torch.tensor(np.random.choice(self.R.shape[1], int(self.R.shape[1] * self.batch_size), replace = False)).to(device)
            
            # make into one hot
            ind_1 = torch.FloatTensor(mask_1.shape[0], self.G.shape[0]).to(device).zero_().scatter_(1, mask_1[:,None], 1)
            ind_2 = torch.FloatTensor(mask_2.shape[0], self.R.shape[0]).to(device).zero_().scatter_(1, mask_2[:,None], 1)
            ind_g = torch.FloatTensor(mask_g.shape[0], self.G.shape[1]).to(device).zero_().scatter_(1, mask_g[:,None], 1)
            ind_r = torch.FloatTensor(mask_r.shape[0], self.R.shape[1]).to(device).zero_().scatter_(1, mask_r[:,None], 1)
            
#             print(torch.sum(ind_1[:, mask_1] - torch.eye(mask_1.shape[0]).cuda()))
#             print(torch.sum(ind_2[:, mask_2] - torch.eye(mask_2.shape[0]).cuda()))
#             print(torch.sum(ind_g[:, mask_g] - torch.eye(mask_g.shape[0]).cuda()))
#             print(torch.sum(ind_r[:, mask_r] - torch.eye(mask_r.shape[0]).cuda()))
            
            # make into matrix rather than vector, think
            out1, out2, embed_u1, embed_v1, embed_u2, embed_v2 = self.ncf(u1 = ind_1, u2 = ind_2, v1 = ind_g, v2 = ind_r)

            loss_rna = self.recon_loss(x_est = out1, x = self.G[mask_1,:][:, mask_g], mode = "RNA") 
            loss_atac = self.recon_loss(x_est = out2, x = self.R[mask_2,:][:, mask_r], mode = "ATAC")
            loss_gact = (self.ncf.gene_act - self.A).pow(2).mean()
            loss_entropy = torch.sum(torch.abs(embed_u1)) + torch.sum(torch.abs(embed_u2))#(self.entropy_loss(embed_u1) + self.entropy_loss(embed_u2))
            
            loss = self.alpha[0] * loss_rna + self.alpha[1] * loss_atac + self.alpha[2] * loss_gact + self.alpha[3] * loss_entropy
            loss.backward()
            self.optimizer.step()
                
            if (t+1) % self.interval == 0:
                print('Epoch {}, Validating Loss: {:.4f}'.format(t + 1, loss.item()))
                info = [
                    'loss RNA: {:.5f}'.format(loss_rna.item()),
                    'loss ATAC: {:.5f}'.format(loss_atac.item()),
                    'loss gact: {:.5f}'.format(loss_gact.item()),
                    'loss entropy: {:.5f}'.format(loss_entropy.item())
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
                        if self.optimizer.param_groups[0]['lr'] < 1e-5:
                            break
                        else:
                            self.load_state_dict(torch.load(f'../check_points/real_{self.N}.pt'))
                            count = 0

