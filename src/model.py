import sys, os
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as opt
import pandas as pd
from torch.nn import Module, Parameter
from torch import softmax, log_softmax, Tensor
from torch_sparse import SparseTensor
from sklearn.cluster import KMeans
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import time
from utils import preprocess, csr2st


class cfrmModel(Module):
    def __init__(self, dir, N=3, batch_size=100, lr=1e-3, dropout=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.alpha = torch.FloatTensor([1, 1, 1, 1, 1, 1])
        self.dropout = dropout
        # data
        counts_rna = np.loadtxt(os.path.join(dir, 'GxC1.txt')).T
        counts_atac = np.loadtxt(os.path.join(dir, 'RxC2.txt')).T

        counts_rna = preprocess(counts=counts_rna, mode="quantile", modality="RNA")
        counts_atac = preprocess(counts=counts_atac, mode="quantile", modality="ATAC")

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
            loss4 = self.C_1[mask_1].abs().mean() + self.C_2[mask_2].abs().mean()
            loss3 = 0
        elif mode == 'C_gr':
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.C_1[mask_1].detach() @ (self.A_1g_l * self.A * self.A_1g_r).detach() @ self.C_g[mask_g].t()).pow(2).mean()
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.C_2[mask_2].detach() @ (self.A_2r_l * self.A * self.A_2r_r).detach() @ self.C_r[mask_r].t()).pow(2).mean()
            loss3 = (self.M[np.ix_(mask_g, mask_r)] - self.C_g[mask_g] @ (self.A_gr_l * self.A * self.A_gr_r).detach() @ self.C_r[mask_r].t()).pow(2).mean()
            loss4 = self.C_g[mask_g].abs().mean() + self.C_r[mask_r].abs().mean()
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
                loss4 = (self.C_1.abs().mean() + self.C_2.abs().mean() + self.C_g.abs().mean() + self.C_r.abs().mean())
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
                    'loss sparse: {:.5f}'.format(loss4.item()),
                ]
                for i in info:
                    print("\t", i)
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(self.state_dict(), '../check_points/real.pt')
                count = 0
            else:
                count += 1
                if count % 20 == 0:
                    self.optimizer.param_groups[0]['lr'] *= 0.5
                    print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                    if self.optimizer.param_groups[0]['lr'] < 1e-4:
                        break
                    else:
                        self.load_state_dict(torch.load('../check_points/real.pt'))
                        count = 0

class NewModel(Module):
    def __init__(self, dir, N=3, batch_size=1.0, interval=10, lr=1e-3, init='svd'):
        super().__init__()
        self.batch_size = batch_size
        self.interval = interval
        self.alpha = torch.FloatTensor([1, 1, 1, 1, 1, 1])
        # data
        counts_rna = sp.load_npz(os.path.join(dir, 'C1xG.npz'))
        counts_atac = sp.load_npz(os.path.join(dir, 'C2xR.npz')).astype(np.float32)
        A = sp.load_npz(os.path.join(dir, 'GxR.npz')).tocoo()

        self.G = torch.FloatTensor(counts_rna.todense())
        self.R = torch.FloatTensor(counts_atac.todense())
        self.A = torch.FloatTensor(A.todense())

        self.s_g = Parameter(torch.ones(1, self.G.shape[1]))
        self.s_r = Parameter(torch.ones(1, self.R.shape[1]))
        self.b_g = Parameter(torch.zeros(1, self.G.shape[1]))
        self.b_r = Parameter(torch.zeros(1, self.R.shape[1]))
        self.C_1 = Parameter(torch.randn(self.G.shape[0], N))
        self.C_2 = Parameter(torch.randn(self.R.shape[0], N))
        self.C_g = Parameter(torch.randn(self.G.shape[1], N))
        self.C_r = Parameter(torch.randn(self.R.shape[1], N))

        self.meta_rna = pd.read_csv(os.path.join(dir, "meta_rna.csv"), index_col=0)
        self.meta_atac = pd.read_csv(os.path.join(dir, "meta_atac.csv"), index_col=0)
        self.regions = pd.read_csv(os.path.join(dir, "regions.txt"), header=None)
        self.genes = pd.read_csv(os.path.join(dir, "genes.txt"), header=None)

        self.optimizer = opt.Adam(self.parameters(), lr=lr)

        # with torch.no_grad():
        #     loss, *_ = self.batch_loss('valid')
        #     print('Initial Loss is {:.5f}'.format(loss.item()))

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
            loss3, loss4 = 0, 0
        elif mode == 'C_gr':
            loss1 = (self.G[np.ix_(mask_1, mask_g)] - self.s_g[:, mask_g] * (self.softmax(self.C_1[mask_1].detach()) @
                     self.softmax(self.C_g[mask_g]).t()) - self.b_g[:, mask_g]).pow(2).mean()
            loss2 = (self.R[np.ix_(mask_2, mask_r)] - self.s_r[:, mask_r] * (self.softmax(self.C_2[mask_2].detach()) @
                     self.softmax(self.C_r[mask_r]).t()) - self.b_r[:, mask_r]).pow(2).mean()
            loss3 = (self.A[mask_g] @ self.C_r - self.C_g[mask_g]).pow(2).mean()
            loss4 = self.entropy_loss(self.C_g[mask_g]) + self.entropy_loss(self.C_r[mask_g])
        elif mode == 'valid':
            with torch.no_grad():
                loss1 = (self.G - self.s_g * (self.softmax(self.C_1) @
                         self.softmax(self.C_g).t()) - self.b_g).pow(2).mean()
                loss2 = (self.R - self.s_r * (self.softmax(self.C_2.detach()) @
                         self.softmax(self.C_r).t()) - self.b_r).pow(2).mean()
                loss3 = (self.A @ self.C_r - self.C_g).pow(2).mean()
                loss4 = self.entropy_loss(self.C_g) + self.entropy_loss(self.C_r)
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
            for mode in ['C_12', 'C_gr', 'C_12', 'C_gr']:
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
                    'loss sparse: {:.5f}'.format(loss4.item()),
                ]
                for i in info:
                    print("\t", i)
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(self.state_dict(), '../check_points/real.pt')
                count = 0
            else:
                count += 1
                if count % 20 == 0:
                    self.optimizer.param_groups[0]['lr'] *= 0.5
                    print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                    if self.optimizer.param_groups[0]['lr'] < 1e-4:
                        break
                    else:
                        self.load_state_dict(torch.load('../check_points/real.pt'))
                        count = 0

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = NewModel(dir = '../data/real/BMMC/', N=25, lr=1e-3, interval=1, batch_size=0.2).to(device)
    model.train_func(T=10000)
