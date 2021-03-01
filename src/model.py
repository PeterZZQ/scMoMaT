import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as opt
from torch.nn import Module, Parameter
from sklearn.cluster import KMeans

class cfrmModel(Module):
    def __init__(self, dir, N=3, batch_size=100, lr=1e-3, dropout=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.alpha = torch.FloatTensor([1, 1, 1, 1, 1])
        self.dropout = dropout
        # data
        self.G = torch.FloatTensor(np.loadtxt(os.path.join(dir, 'GxC1.txt'))).t()
        self.R = torch.FloatTensor(np.loadtxt(os.path.join(dir, 'RxC2.txt'))).t()
        self.A = torch.FloatTensor(np.loadtxt(os.path.join(dir, 'RxG.txt')))
        assert self.A.shape[0] == self.R.shape[1]
        assert self.A.shape[1] == self.G.shape[1]
        self.label_g = torch.LongTensor(np.loadtxt(os.path.join(dir, 'gene_label.txt')))
        self.label_c1 = torch.LongTensor(np.loadtxt(os.path.join(dir, 'cell_label_C1.txt'), skiprows=1, usecols=[1]))
        self.label_c2 = torch.LongTensor(np.loadtxt(os.path.join(dir, 'cell_label_C2.txt'), skiprows=1, usecols=[1]))
        # learnable parameters
        self.A_1g = Parameter(torch.rand(N, N))
        self.A_2r = Parameter(torch.rand(N, N))
        self.D_gr = Parameter(torch.rand(N, 1))
        self.D_12 = Parameter(torch.rand(N, 1))
        u, s, v = torch.svd(self.G)
        self.C_1g = Parameter(torch.FloatTensor(u[:, :N]))
        self.alpha[0] = 1 / s[0]
        u, s, v = torch.svd(self.R)
        self.C_2r = Parameter(torch.FloatTensor(u[:, :N]))
        self.alpha[1] = 1 / s[0]
        u, s, v = torch.svd(self.A)
        self.C_r = Parameter(torch.FloatTensor(u[:, :N]))
        self.C_g = Parameter(torch.FloatTensor(v[:, :N]))
        self.alpha[2] = 1 / s[0]
        # optimizer
        self.optimizer = opt.Adam(self.parameters(), lr=lr)

    @staticmethod
    def orthogonal_loss(A):
        return (A.t() @ A - torch.eye(A.shape[1])).pow(2).sum()

    def batch_loss(self):
        print(self.C_1g[:10])
        loss1 = (self.G - self.C_1g @ self.A_1g @ self.C_g.t()).abs().sum()
        loss2 = (self.R - self.C_2r @ self.A_2r @ self.C_r.t()).abs().sum()
        loss3 = (self.A - self.C_r @ (self.D_gr * self.C_g.t())).abs().sum()
        loss4 = (self.A_1g - self.D_12 * self.A_2r).pow(2).mean()
        loss5 = sum(map(self.orthogonal_loss, [self.C_1g, self.C_2r, self.C_g, self.C_r]))
        print(loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item())
        loss = self.alpha[0] * loss1 + self.alpha[1] * loss2 + self.alpha[2] * loss3 + self.alpha[3] * loss4 + self.alpha[4] * loss5
        return loss

    def train_func(self, T):
        best_loss = 1e8
        count = 0
        for t in range(T):
            self.optimizer.zero_grad()
            loss = self.batch_loss()
            loss.backward()
            self.optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                count = 0
            else:
                count += 1
                if count % 20 == 0:
                    self.optimizer.param_groups[0]['lr'] *= 0.5
                    print('Epoch: {}, shrink lr to {:.4f}'.format(t+1, self.optimizer.param_groups[0]['lr']))
                    if self.optimizer.param_groups[0]['lr'] < 1e-4:
                        break
                    else:
                        count = 0
            print('Epoch {}, Training Loss: {:.4f}'.format(t+1, loss.item()))




if __name__ == '__main__':
    model = cfrmModel(dir = '../data/simulated/2batches_3clusts', N=3, dropout=0)
    model.train_func(T=10000)
    torch.save(model.state_dict(), '../check_points/simulated_2b_3c.pt')