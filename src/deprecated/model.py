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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    


class cfrm_goodinit(Module):
    def __init__(self, counts, N=3, batch_size=1.0, interval=10, lr=1e-3, alpha = [1000, 1000, 100, 0.0, 0.0], seed = None):
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
        
        Ainit = torch.randn((self.N1,self.N2))
        self.A_r = Parameter(Ainit)        
        self.A_g = Parameter(Ainit)

        self.C_1 = Parameter(torch.rand(self.G.shape[0], self.N1))
        self.C_2 = Parameter(torch.rand(self.R.shape[0], self.N1))
        self.C_r = Parameter(torch.rand(self.R.shape[1], self.N2))
        self.C_g = Parameter(torch.rand(self.G.shape[1], self.N2))

        self.b_g = torch.zeros(1, self.G.shape[1]).to(device)
        self.b_r = torch.zeros(1, self.R.shape[1]).to(device)
        self.b_1 = torch.zeros(self.G.shape[0],1).to(device)
        self.b_2 = torch.zeros(self.R.shape[0],1).to(device)
        
        self.optimizer = opt.Adam(self.parameters(), lr=lr)
        

    
    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim = 1)

    
    def train_func(self, T, match = False):
        # train R and G separately
        for t in range(T):

            mask_1 = np.random.choice(self.G.shape[0], int(self.G.shape[0] * self.batch_size), replace=False)
            mask_g = np.random.choice(self.G.shape[1], int(self.G.shape[1] * self.batch_size), replace=False)

            mask_2 = np.random.choice(self.R.shape[0], int(self.R.shape[0] * self.batch_size), replace=False)
            mask_r = np.random.choice(self.R.shape[1], int(self.R.shape[1] * self.batch_size), replace=False)

            for mode in ["Ab"] * 1 + ["C_12"] * 10 + ["C_gr"] * 10:
                if mode == "Ab":
                    with torch.no_grad():
                        M = self.softmax(self.C_1[mask_1,:]).t() @ (self.G[np.ix_(mask_1, mask_g)] - self.b_g[:, mask_g] - self.b_1[mask_1,:]) @ self.softmax(self.C_g[mask_g,:])
                        A_g = torch.inverse(self.softmax(self.C_1[mask_1,:]).t() @ self.softmax(self.C_1[mask_1,:])) @ M @ torch.inverse(self.softmax(self.C_g[mask_g,:]).t() @ self.softmax(self.C_g[mask_g,:]))

                        M = self.softmax(self.C_2[mask_2,:]).t() @ (self.R[np.ix_(mask_2, mask_r)] - self.b_r[:, mask_r] - self.b_2[mask_2,:]) @ self.softmax(self.C_r[mask_r,:])
                        A_r = torch.inverse(self.softmax(self.C_2[mask_2,:]).t() @ self.softmax(self.C_2[mask_2,:])) @ M @ torch.inverse(self.softmax(self.C_r[mask_r,:]).t() @ self.softmax(self.C_r[mask_r,:]))

                        self.b_g[:, mask_g] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ A_g @ self.softmax(self.C_g)[mask_g,:].t() - self.b_1[mask_1,:], dim = 0)[None,:]/mask_1.shape[0]
                        self.b_r[:, mask_r] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ A_r @ self.softmax(self.C_r[mask_r, :]).t() - self.b_2[mask_2,:], dim = 0)[None,:]/mask_2.shape[0]

                        self.b_1[mask_1, :] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ A_g @  self.softmax(self.C_g)[mask_g,:].t() - self.b_g[:, mask_g], dim = 1)[:,None]/mask_g.shape[0]
                        self.b_2[mask_2, :] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ A_r @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r], dim = 1)[:,None]/mask_r.shape[0]                        


                elif mode == "C_12":
                    loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ A_g @ self.softmax(self.C_g[mask_g, :].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

                    loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ A_r @ self.softmax(self.C_r[mask_r,:].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()

                    loss1.backward()
                    loss2.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                elif mode == "C_gr":
                    loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]).detach() @ A_g @ self.softmax(self.C_g[mask_g, :]).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

                    loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]).detach() @ A_r @ self.softmax(self.C_r[mask_r,:]).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()

                    loss1.backward()
                    loss2.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()   


            if (t%100 == 0):
                with torch.no_grad():
                    loss1 = (self.G - self.softmax(self.C_1) @ A_g @ self.softmax(self.C_g).t() - self.b_g - self.b_1).pow(2).mean()

                    loss2 = (self.R - self.softmax(self.C_2) @ A_r @ self.softmax(self.C_r).t() - self.b_r - self.b_2).pow(2).mean()
                print("iteration: " + str(t) + ", loss1: " + str(loss1.item()) + ", loss2: " + str(loss2.item()))

        
        with torch.no_grad():
            self.A_g.data = A_g
            self.A_r.data = A_r
        
        if match:
            with torch.no_grad():
                # match dimensions
                perm2 = list(itertools.permutations(range(self.N2)))
                score2 = np.zeros(len(perm2))
                for i in range(len(perm2)):
                    score2[i] = torch.trace(self.C_g[:,perm2[i]].t() @ self.A @ self.C_r) /torch.norm(self.softmax(self.C_g)) / torch.norm(self.A @ self.softmax(self.C_r))

                match2 = np.argmax(score2)
                print(perm2[match2])

                perm1 = list(itertools.permutations(range(self.N1)))
                score1 = np.zeros(len(perm1))
                for i in range(len(perm1)):
                    score1[i] = torch.trace(A_g[perm1[i],:].t() @ A_r) /torch.norm(A_g) / torch.norm(A_r)

                match1 = np.argmax(score1)
                print(perm1[match1])

                # assign permuated values
                self.C_g.data = self.C_g.data[:,perm2[match2]]
                A_g = A_g[:, perm2[match2]]

                self.A_g.data = A_g[perm1[match1], :]
                self.A_r.data = A_r
                self.C_1.data = self.C_1.data[:, perm1[match1]]
       
        return self.C_1.data, self.C_2.data, self.A_g.data, self.A_r.data, self.C_g.data, self.C_r.data


class cfrm_diaginit(Module):
    def __init__(self, counts, N=3, batch_size=1.0, interval=10, lr=1e-3, alpha = [1000, 1000, 100, 0.0, 0.0], seed = None):
        super().__init__()
        self.N = N
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
        
        # diagonal value
        self.A_r = Parameter(torch.eye(self.N))        
        self.A_g = Parameter(torch.eye(self.N))

        self.C_1 = Parameter(torch.rand(self.G.shape[0], self.N))
        self.C_2 = Parameter(torch.rand(self.R.shape[0], self.N))
        self.C_r = Parameter(torch.rand(self.R.shape[1], self.N))
        self.C_g = Parameter(torch.rand(self.G.shape[1], self.N))

        self.b_g = torch.zeros(1, self.G.shape[1]).to(device)
        self.b_r = torch.zeros(1, self.R.shape[1]).to(device)
        self.b_1 = torch.zeros(self.G.shape[0],1).to(device)
        self.b_2 = torch.zeros(self.R.shape[0],1).to(device)
        
        self.optimizer = opt.Adam(self.parameters(), lr=lr)
        

    
    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim = 1)

    
    def train_func(self, T, match = False):
        # train R and G separately
        A_g = torch.diag(self.A_g.data)
        A_r = torch.diag(self.A_r.data)
        for t in range(T):

            mask_1 = np.random.choice(self.G.shape[0], int(self.G.shape[0] * self.batch_size), replace=False)
            mask_g = np.random.choice(self.G.shape[1], int(self.G.shape[1] * self.batch_size), replace=False)

            mask_2 = np.random.choice(self.R.shape[0], int(self.R.shape[0] * self.batch_size), replace=False)
            mask_r = np.random.choice(self.R.shape[1], int(self.R.shape[1] * self.batch_size), replace=False)

            for mode in ["Ab"] * 1 + ["C_12"] * 10 + ["C_gr"] * 10:
                if mode == "Ab":
                    with torch.no_grad():
                        
                        # update A_g
                        G_temp = (self.G[np.ix_(mask_1, mask_g)] - self.b_g[:, mask_g] - self.b_1[mask_1,:]).reshape(-1, 1)
                        M = torch.zeros((G_temp.shape[0], self.N)).to(device)
                        for i in range(self.N):
                            M[:,i] = (self.softmax(self.C_1)[mask_1,i:i+1] @ self.softmax(self.C_g)[mask_g,i:i+1].t()).reshape(-1)
                        
                        A_g = (torch.inverse(M.t() @ M) @ M.t() @ G_temp).squeeze()
                        
                        
                        # update A_r
                        R_temp = (self.R[np.ix_(mask_2, mask_r)] - self.b_r[:, mask_r] - self.b_2[mask_2,:]).reshape(-1, 1)
                        M = torch.zeros((R_temp.shape[0], self.N)).to(device)
                        for i in range(self.N):
                            M[:,i] = (self.softmax(self.C_2)[mask_2,i:i+1] @ self.softmax(self.C_r)[mask_r,i:i+1].t()).reshape(-1)

                        A_r = (torch.inverse(M.t() @ M) @ M.t() @ R_temp).squeeze()
                        # keep the same
                        self.b_g[:, mask_g] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ torch.diag(A_g) @ self.softmax(self.C_g)[mask_g,:].t() - self.b_1[mask_1,:], dim = 0)[None,:]/mask_1.shape[0]
                        self.b_r[:, mask_r] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ torch.diag(A_r) @ self.softmax(self.C_r[mask_r, :]).t() - self.b_2[mask_2,:], dim = 0)[None,:]/mask_2.shape[0]

                        self.b_1[mask_1, :] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ torch.diag(A_g) @  self.softmax(self.C_g)[mask_g,:].t() - self.b_g[:, mask_g], dim = 1)[:,None]/mask_g.shape[0]
                        self.b_2[mask_2, :] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ torch.diag(A_r) @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r], dim = 1)[:,None]/mask_r.shape[0]                        


                elif mode == "C_12":
                    loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ torch.diag(A_g) @ self.softmax(self.C_g[mask_g, :].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

                    loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ torch.diag(A_r) @ self.softmax(self.C_r[mask_r,:].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()

                    loss1.backward()
                    loss2.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                elif mode == "C_gr":
                    loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]).detach() @ torch.diag(A_g) @ self.softmax(self.C_g[mask_g, :]).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

                    loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]).detach() @ torch.diag(A_r) @ self.softmax(self.C_r[mask_r,:]).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()

                    loss1.backward()
                    loss2.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()   


            if (t%100 == 0):
                with torch.no_grad():
                    loss1 = (self.G - self.softmax(self.C_1) @ torch.diag(A_g) @ self.softmax(self.C_g).t() - self.b_g - self.b_1).pow(2).mean()

                    loss2 = (self.R - self.softmax(self.C_2) @ torch.diag(A_r) @ self.softmax(self.C_r).t() - self.b_r - self.b_2).pow(2).mean()
                print("iteration: " + str(t) + ", loss1: " + str(loss1.item()) + ", loss2: " + str(loss2.item()))

        
        with torch.no_grad():
            self.A_g.data = torch.diag(A_g)
            self.A_r.data = torch.diag(A_r)
        
        if match:
            with torch.no_grad():
                A_g = torch.diag(A_g)
                A_r = torch.diag(A_r)
                # match dimensions
                perm2 = list(itertools.permutations(range(self.N)))
                score2 = np.zeros(len(perm2))
                for i in range(len(perm2)):
                    score2[i] = torch.trace(self.C_g[:,perm2[i]].t() @ self.A @ self.C_r) /torch.norm(self.softmax(self.C_g)) / torch.norm(self.A @ self.softmax(self.C_r))

                match2 = np.argmax(score2)
                print(perm2[match2])

                perm1 = list(itertools.permutations(range(self.N)))
                score1 = np.zeros(len(perm1))
                for i in range(len(perm1)):
                    score1[i] = torch.trace(A_g[perm1[i],:].t() @ A_r) /torch.norm(A_g) / torch.norm(A_r)

                match1 = np.argmax(score1)
                print(perm1[match1])

                # assign permuated values
                self.C_g.data = self.C_g.data[:,perm2[match2]]
                A_g = A_g[:, perm2[match2]]

                self.A_g.data = A_g[perm1[match1], :]
                self.A_r.data = A_r
                self.C_1.data = self.C_1.data[:, perm1[match1]]
       
        return self.C_1.data, self.C_2.data, self.A_g.data, self.A_r.data, self.C_g.data, self.C_r.data


class cfrm_nmfinit(Module):
    def __init__(self, counts, N=3, batch_size=1.0, lr=1e-3, seed = None):
        super().__init__()
        self.N = N
        self.batch_size = batch_size
        
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

        self.C_1 = Parameter(torch.rand(self.G.shape[0], self.N))
        self.C_2 = Parameter(torch.rand(self.R.shape[0], self.N))
        self.C_r = Parameter(torch.rand(self.R.shape[1], self.N))
        self.C_g = Parameter(torch.rand(self.G.shape[1], self.N))

        self.b_g = torch.zeros(1, self.G.shape[1]).to(device)
        self.b_r = torch.zeros(1, self.R.shape[1]).to(device)
        self.b_1 = torch.zeros(self.G.shape[0],1).to(device)
        self.b_2 = torch.zeros(self.R.shape[0],1).to(device)
        
        self.optimizer = opt.Adam(self.parameters(), lr=lr)
        

    
    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim = 1)

    
    def train_func(self, T, match = False):
        # train R and G separately
        for t in range(T):

            mask_1 = np.random.choice(self.G.shape[0], int(self.G.shape[0] * self.batch_size), replace=False)
            mask_g = np.random.choice(self.G.shape[1], int(self.G.shape[1] * self.batch_size), replace=False)

            mask_2 = np.random.choice(self.R.shape[0], int(self.R.shape[0] * self.batch_size), replace=False)
            mask_r = np.random.choice(self.R.shape[1], int(self.R.shape[1] * self.batch_size), replace=False)

            for mode in ["b"] * 1 + ["C_12"] * 10 + ["C_gr"] * 10:
                if mode == "b":
                    with torch.no_grad():
                        # keep the same
                        self.b_g[:, mask_g] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.softmax(self.C_g)[mask_g,:].t() - self.b_1[mask_1,:], dim = 0)[None,:]/mask_1.shape[0]
                        self.b_r[:, mask_r] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ self.softmax(self.C_r[mask_r, :]).t() - self.b_2[mask_2,:], dim = 0)[None,:]/mask_2.shape[0]

                        self.b_1[mask_1, :] = torch.sum(self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.softmax(self.C_g)[mask_g,:].t() - self.b_g[:, mask_g], dim = 1)[:,None]/mask_g.shape[0]
                        self.b_2[mask_2, :] = torch.sum(self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r], dim = 1)[:,None]/mask_r.shape[0]                        


                elif mode == "C_12":
                    loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]) @ self.softmax(self.C_g[mask_g, :].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

                    loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]) @ self.softmax(self.C_r[mask_r,:].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()

                    loss1.backward()
                    loss2.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                elif mode == "C_gr":
                    loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:]).detach() @ self.softmax(self.C_g[mask_g, :]).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

                    loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:]).detach() @ self.softmax(self.C_r[mask_r,:]).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()

                    loss1.backward()
                    loss2.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()   


            if (t%T == 0):
                with torch.no_grad():
                    loss1 = (self.G - self.softmax(self.C_1) @ self.softmax(self.C_g).t() - self.b_g - self.b_1).pow(2).mean()

                    loss2 = (self.R - self.softmax(self.C_2) @ self.softmax(self.C_r).t() - self.b_r - self.b_2).pow(2).mean()
                print("iteration: " + str(t) + ", loss1: " + str(loss1.item()) + ", loss2: " + str(loss2.item()))

        
        if match:
            with torch.no_grad():
                # match dimensions
                perm = list(itertools.permutations(range(self.N)))
                score = np.zeros(len(perm))
                for i in range(len(perm)):
                    score[i] = torch.trace(self.C_g[:,perm[i]].t() @ self.A @ self.C_r) /torch.norm(self.softmax(self.C_g)) / torch.norm(self.A @ self.softmax(self.C_r))

                match = np.argmax(score)
               
                # assign permuated values
                self.C_g.data = self.C_g.data[:,perm[match]]
                self.C_1.data = self.C_1.data[:, perm[match]]
       
        return self.C_1.data, self.C_2.data, torch.eye(self.N), torch.eye(self.N), self.C_g.data, self.C_r.data


class cfrm_svdinit(Module):
    def __init__(self, counts, N=3):
        super().__init__()
        self.N1 = N
        self.N2 = N
        
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

    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim = 1)
       
    def train_func(self, match = False):
        with torch.no_grad():
            # svd
            U, S, V = torch.svd(self.G)
            C_1 = U[:,:self.N1]
            A_g = torch.diag(S[:self.N1])
            C_g = V[:,:self.N2]

            U, S, V = torch.svd(self.R)
            C_2 = U[:,:self.N1]
            A_r = torch.diag(S[:self.N1])
            C_r = V[:,:self.N2]
            
            # match dimensions
            if match:
                perm = list(itertools.permutations(range(self.N2)))
                score = np.zeros(len(perm))
                for i in range(len(perm)):
                    score[i] = torch.trace(C_g[:,perm[i]].t() @ self.A @ C_r) /torch.norm(self.softmax(C_g)) / torch.norm(self.A @ self.softmax(C_r))

                match = np.argmax(score)

                C_g = C_g[:, perm[match]]
                A_g = A_g[np.ix_(perm[match], perm[match])]
                C_1 = C_1[:, perm[match]]
            
        return C_1, C_2, A_g, A_r, C_g, C_r

    

    
######################################################################################################################            
class cfrm_new(Module):
    def __init__(self, counts, N=3, batch_size=1.0, interval=10, lr=1e-3, alpha = [1000, 1000, 100, 0.0, 0.0], seed = None, init = None):
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
            
                
        elif mode == 'C_gr':
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g.detach() @ self.softmax(self.C_g[mask_g, :]).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = 0 
            loss4 = - torch.trace(self.softmax(self.C_g[mask_g,:]).t() @ self.A[np.ix_(mask_g, mask_r)] @ self.softmax(self.C_r[mask_r,:])) /torch.norm(self.softmax(self.C_g[mask_g,:])) / torch.norm(self.A[np.ix_(mask_g, mask_r)] @ self.softmax(self.C_r[mask_r,:]))
             
            
        
        elif mode == "A":
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g @  self.softmax(self.C_g[mask_g,:].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r @ self.softmax(self.C_r[mask_r, :].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = - torch.trace(self.A_r.t() @ self.A_g)/torch.norm(self.A_r)/torch.norm(self.A_g)
            loss4 = 0
       
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

            
        elif mode == 'valid':
            with torch.no_grad():
                loss1 = (self.G - self.softmax(self.C_1) @ self.A_g @ self.softmax(self.C_g).t() - self.b_g - self.b_1).pow(2).mean()
                     
                loss2 = (self.R - self.softmax(self.C_2) @ self.A_r @ self.softmax(self.C_r).t() - self.b_r - self.b_2).pow(2).mean()
                 
                loss3 = - torch.trace(self.A_r.t() @ self.A_g)/torch.norm(self.A_r)/torch.norm(self.A_g)
                loss4 = - torch.trace(self.softmax(self.C_g).t() @ self.A @ self.softmax(self.C_r)) /torch.norm(self.softmax(self.C_g)) / torch.norm(self.A @ self.softmax(self.C_r))
                     
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
            
            for mode in ['b']* 1 + ['C_12']*1 + ['C_gr']*1 + ['A']*1:
                loss, loss1, loss2, loss3, _ = self.batch_loss(mode)
                
                if mode != 'b':
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                ###########################################################
                '''
                loss, loss1, loss2, loss3, loss4 = self.batch_loss('valid')
                if (mode == 'C_12'):
                    losses1.append(loss1)
                if (mode == 'C_gr'):
                    losses2.append(loss2)
                if (mode == 'A'):
                    losses3.append(loss3)
                    
            if (t%self.interval == 0):
                losses1 = np.array(losses1)
                fig = plt.figure(figsize = (5,5))
                ax = fig.add_subplot()
                ax.plot(np.arange(losses1.shape[0]), losses1)
                ax.set_title("iter " + str(t) + ", loss: C_12")

                losses2 = np.array(losses2)
                fig = plt.figure(figsize = (5,5))
                ax = fig.add_subplot()
                ax.plot(np.arange(losses2.shape[0]), losses2)
                ax.set_title("iter " + str(t) + ", loss: C_rg")
                
                losses3 = np.array(losses3)
                fig = plt.figure(figsize = (5,5))
                ax = fig.add_subplot()
                ax.plot(np.arange(losses3.shape[0]), losses3)
                ax.set_title("iter " + str(t) + ", loss: A")
                '''

                ###########################################################
                
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

class cfrm(Module):
    def __init__(self, counts, N=3, batch_size=1.0, interval=10, lr=1e-3, alpha = [1000, 1000, 100, 0.0, 0.0], seed = None, init = None):
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


        self.b_g = Parameter(torch.zeros(1, self.G.shape[1]))
        self.b_r = Parameter(torch.zeros(1, self.R.shape[1]))
        self.b_1 = Parameter(torch.zeros(self.G.shape[0],1))
        self.b_2 = Parameter(torch.zeros(self.R.shape[0],1))
        
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
            
                
        elif mode == 'C_gr':
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g.detach() @ self.softmax(self.C_g[mask_g, :]).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r, :]).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = 0 
            loss4 = - torch.trace(self.softmax(self.C_g[mask_g,:]).t() @ self.A[np.ix_(mask_g, mask_r)] @ self.softmax(self.C_r[mask_r,:])) /torch.norm(self.softmax(self.C_g[mask_g,:])) / torch.norm(self.A[np.ix_(mask_g, mask_r)] @ self.softmax(self.C_r[mask_r,:]))
             
            
        
        elif mode == "A":
            
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g @  self.softmax(self.C_g[mask_g,:].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()
            
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r @ self.softmax(self.C_r[mask_r, :].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            
            loss3 = - torch.trace(self.A_r.t() @ self.A_g)/torch.norm(self.A_r)/torch.norm(self.A_g)
            loss4 = 0
       
        elif mode == "b":

            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.softmax(self.C_1[mask_1,:].detach()) @ self.A_g.detach() @ self.softmax(self.C_g[mask_g, :].detach()).t() - self.b_g[:, mask_g] - self.b_1[mask_1,:]).pow(2).mean()

            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.softmax(self.C_2[mask_2,:].detach()) @ self.A_r.detach() @ self.softmax(self.C_r[mask_r, :].detach()).t() - self.b_r[:, mask_r] - self.b_2[mask_2,:]).pow(2).mean()
            loss3 = 0
            loss4 = 0
            
        elif mode == 'valid':
            with torch.no_grad():
                loss1 = (self.G - self.softmax(self.C_1) @ self.A_g @ self.softmax(self.C_g).t() - self.b_g - self.b_1).pow(2).mean()
                     
                loss2 = (self.R - self.softmax(self.C_2) @ self.A_r @ self.softmax(self.C_r).t() - self.b_r - self.b_2).pow(2).mean()
                 
                loss3 = - torch.trace(self.A_r.t() @ self.A_g)/torch.norm(self.A_r)/torch.norm(self.A_g)
                loss4 = - torch.trace(self.softmax(self.C_g).t() @ self.A @ self.softmax(self.C_r)) /torch.norm(self.softmax(self.C_g)) / torch.norm(self.A @ self.softmax(self.C_r))
                     
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
            
            for mode in ['b']* 1 + ['C_12']*1 + ['C_gr']*1 + ['A']*1:
                loss, loss1, loss2, loss3, _ = self.batch_loss(mode)
                
                # if mode != 'b':
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


                            