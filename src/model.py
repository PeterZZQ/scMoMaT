import sys, os
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as opt
from torch.nn import Module, Parameter
from sklearn.cluster import KMeans
from utils import preprocess


class cfrmModel(Module):
    def __init__(self, dir, N=3, batch_size=100, lr=1e-3, dropout=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.alpha = torch.FloatTensor([1, 1, 1, 1, 10])
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
        loss = self.batch_loss('valid')
        for l in loss:
            print(l.item())
        # optimizer
        self.optimizer = opt.Adam(self.parameters(), lr=lr)

    @staticmethod
    def orthogonal_loss(A):
        return (A.t() @ A - torch.eye(A.shape[1])).pow(2).sum()

    def batch_loss(self, mode='C_c'):
        if mode == 'C_c':
            loss1 = (self.G - self.C_1 @ self.A_1g.detach() @ self.C_g.detach().t()).pow(2).mean()
            loss2 = (self.R - self.C_2 @ self.A_2r.detach() @ self.C_r.detach().t()).pow(2).mean()
            loss5 = sum(map(self.orthogonal_loss, [self.C_1, self.C_2]))
            loss3, loss4 = 0, 0
        elif mode == 'C_r':
            loss2 = (self.R - self.C_2.detach() @ self.A_2r.detach() @ self.C_r.t()).pow(2).mean()
            loss3 = (self.A - self.C_r @ (self.D_gr.detach() * self.C_g.detach().t())).pow(2).mean()
            loss5 = sum(map(self.orthogonal_loss, [self.C_r]))
            loss1, loss4 = 0, 0
        elif mode == 'C_g':
            loss1 = (self.G - self.C_1.detach() @ self.A_1g.detach() @ self.C_g.t()).pow(2).mean()
            loss3 = (self.A - self.C_r.detach() @ (self.D_gr.detach() * self.C_g.t())).pow(2).mean()
            loss5 = sum(map(self.orthogonal_loss, [self.C_g]))
            loss2, loss4 = 0, 0
        elif mode == 'valid':
            with torch.no_grad():
                loss1 = (self.G - self.C_1 @ self.A_1g @ self.C_g.t()).pow(2).mean()
                loss2 = (self.R - self.C_2 @ self.A_2r @ self.C_r.t()).pow(2).mean()
                loss3 = (self.A - self.C_r @ (self.D_gr * self.C_g.t())).pow(2).mean()
                loss4 = (self.A_1g - self.D_12 * self.A_2r).pow(2).mean()
                loss5 = sum(map(self.orthogonal_loss, [self.C_1, self.C_2, self.C_g, self.C_r]))
        loss = self.alpha[0] * loss1 + self.alpha[1] * loss2 + self.alpha[2] * loss3 + self.alpha[3] * loss4 + \
               self.alpha[4] * loss5
        # loss1 = (self.G - self.C_1 @ self.A_1g @ self.C_g.t()).pow(2).sum()
        # loss2 = (self.R - self.C_2 @ self.A_2r @ self.C_r.t()).pow(2).sum()
        # loss3 = (self.A - self.C_r @ (self.D_gr * self.C_g.t())).pow(2).sum()
        # loss4 = (self.A_1g - self.D_12 * self.A_2r).pow(2).mean()
        # loss5 = sum(map(self.orthogonal_loss, [self.C_1, self.C_2, self.C_g, self.C_r]))
        # loss = self.alpha[0] * loss1 + self.alpha[1] * loss2 + self.alpha[2] * loss3 + self.alpha[3] * loss4 + self.alpha[4] * loss5
        # return loss
        return loss, self.alpha[0] * loss1, self.alpha[1] * loss2, self.alpha[2] * loss3, self.alpha[3] * loss4, \
               self.alpha[4] * loss5

    def train_func(self, T):
        best_loss = 1e12
        count = 0
        for t in range(T):
            self.optimizer.zero_grad()
            for mode in ['C_c']:
                # for mode in ['C_c', 'C_r', 'C_g']:
                loss, loss1, loss2, loss3, loss4, loss5 = self.batch_loss(mode)
                loss.backward()
                self.optimizer.step()
            self.A_1g = self.C_1.t() @ self.G @ self.C_g
            self.A_2r = self.C_2.t() @ self.R @ self.C_r
            self.D_12 = torch.diag((self.A_1g * self.A_2r).sum(dim=1) / self.A_2r.pow(2).sum(dim=1)).detach()
            loss, loss1, loss2, loss3, loss4, loss5 = self.batch_loss('valid')
            print('Epoch {}, Training Loss: {:.4f}'.format(t + 1, loss.item()))
            info = [
                'loss RNA: {:.5f}'.format(loss1.item()),
                'loss ATAC: {:.5f}'.format(loss2.item()),
                'loss gene act: {:.5f}'.format(loss3.item()),
                'loss merge: {:.5f}'.format(loss4.item()),
                'loss ortho: {:.5f}'.format(loss5.item()),
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



if __name__ == '__main__':
    model = cfrmModel(dir = '../data/simulated/2batches_3clusts', N=3, dropout=0)
    model.train_func(T=10000)
    torch.save(model.state_dict(), '../check_points/simulated_2b_3c.pt')