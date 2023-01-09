import torch
import numpy as np
import torch.optim as opt
from torch.nn import Module, Parameter, ParameterDict
from torch import Tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class scmomat_model(Module):
    """\
        Gene clusters more than cell clusters, force A_r and A_g to be sparse:
        
    """
    def __init__(self, counts, K = 30, batch_size = 0.3, interval = 10, lr = 1e-2, lamb = 0.001, seed = None, device = device):
        super().__init__()
        
        # init parameters, 
        # self.K is the number of latent dimensions
        self.K = K
        # latent dimensions for cells and feats
        self.N_cell = self.K
        self.N_feat = self.K
        # number of batches
        self.nbatches = counts["nbatches"]

        self.batch_size = batch_size
        self.interval = interval
        self.device = device
        self.alpha = [1000, lamb * 1000]
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 1. load count matrices
        self.mods = [mod for mod in counts.keys() if (mod != "feats_name") & (mod != "nbatches")]
        self.Xs = {}
        # mod include: RNA, ATAC, PROTEIN, etc.
        for mod in self.mods:
            self.Xs[mod] = []
            for counts_mod in counts[mod]:
                if counts_mod is not None:
                    self.Xs[mod].append(torch.FloatTensor(counts_mod).to(self.device))
                else:
                    self.Xs[mod].append(None)
        
        
        # name of the features
        if "feats_name" in counts.keys():
            self.feats_name = counts["feats_name"]
        else:
            self.feats_name = None

        # put into sanity check
        self.sanity_check()
        
        
        # 2. create parameters
        self.C_cells = ParameterDict({})
        self.C_feats = ParameterDict({})
        self.A_assos = ParameterDict({})
        self.b_cells = {}
        self.b_feats = {}
        self.scales = {}
        
        # create C_cells
        for batch in range(self.nbatches):
            for mod in self.mods:
                if self.Xs[mod][batch] is not None:
                    self.C_cells[str(batch)] = Parameter(torch.rand((self.Xs[mod][batch].shape[0], self.N_cell), device = self.device))
                    break
        
        # create C_feats, matrices exists for all mods
        for mod in self.mods:
            for batch in range(self.nbatches):
                if self.Xs[mod][batch] is not None:
                    self.C_feats[mod] = Parameter(torch.rand((self.Xs[mod][batch].shape[1], self.N_feat), device = self.device))
                    break
        
        # create A_assos
        self.A_assos["shared"] = Parameter(torch.rand((self.N_cell, self.N_feat), device = self.device))
        for mod in self.mods:
            for batch in range(self.nbatches):
                if self.Xs[mod][batch] is not None:
                    self.A_assos[mod + "_" + str(batch)] = Parameter(torch.zeros((self.N_cell, self.N_feat), device = self.device))

        
        # create bias term
        for mod in self.mods:
            self.b_cells[mod] = {}
            self.b_feats[mod] = {}
            for batch in range(self.nbatches):
                if self.Xs[mod][batch] is not None:
                    self.b_cells[mod][batch] = torch.zeros(self.Xs[mod][batch].shape[0], 1).to(self.device)
                    self.b_feats[mod][batch] = torch.zeros(1, self.Xs[mod][batch].shape[1]).to(self.device)

                self.scales[mod] = torch.FloatTensor([1] * self.nbatches).to(self.device)

        
        self.optimizer = opt.Adam(self.parameters(), lr=lr)
        
    
    def sanity_check(self):
        print("Input sanity check...")
        # number of batches are the same
        n_features = {}
        for mod in self.Xs.keys():
            # No all None modality
            if np.all(np.array([x is None for x in self.Xs[mod]]) == True) == True:
                raise ValueError("Don't have count matrix correspond to " + mod)
            
            if (len(self.Xs[mod]) == self.nbatches) == False:
                raise ValueError("Number of batches not match for " + mod)
            
            # feature dimension should be the same
            n_features[mod] = [x.shape[1] for x in self.Xs[mod] if x is not None]
            if np.all(np.array(n_features[mod]) == n_features[mod][0]) == False:
                raise ValueError("Number of features not match for modality " + mod)

            # number of feats_name equals to feature dimension
            if self.feats_name is not None:
                if self.feats_name[mod].shape[0] != n_features[mod][0]:
                    raise ValueError("Feature names do not match the number of features for modality " + mod)

        for batch in range(self.nbatches):
            # cell number of each batch should be the same
            n_cells = np.array([self.Xs[mod][batch].shape[0] for mod in self.Xs.keys() if self.Xs[mod][batch] is not None])
            if len(n_cells) > 1:
                if np.all(n_cells == n_cells[0]) == False:
                    raise ValueError("Number of cells not match between modalities for batch " + str(batch))
            
            # No all None batch
            if np.all(np.array([self.Xs[mod][batch] is None for mod in self.mods]) == True) == True:
                raise ValueError("Don't have count matrix correspond to " + str(batch))    
            
        print("Finished.")
        
    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim = 1)
    
    @staticmethod
    def recon_loss(X, C1, C2, Sigma, b1, b2):
        return (X - C1 @ Sigma @ C2.t() - b1 - b2).pow(2).mean()

    @staticmethod
    def cosine_loss(A, B):
        return -torch.trace(A.t() @ B)/(torch.norm(A) + 1e-6)/(torch.norm(B) + 1e-6)    

    def sample_mini_batch(self):
        """\
        Sample mini batch
        """
        mask_cells = []
        mask_feats = []
        # sample mini_batch for each cell dimension
        for batch in range(self.nbatches):
            mask_cells.append(np.random.choice(self.C_cells[str(batch)].shape[0], int(self.C_cells[str(batch)].shape[0] * self.batch_size), replace=False))
        
        # sample mini_batch for each feature dimension
        for mod in self.mods:
            mask_feats.append(np.random.choice(self.C_feats[mod].shape[0], int(self.C_feats[mod].shape[0] * self.batch_size), replace=False))
        
        return mask_cells, mask_feats

    def batch_loss(self, mode, alpha, batch_indices = None):
        """\
            Calculate overall loss term
        """
        # init
        loss1 = torch.FloatTensor([0]).to(self.device)
        loss2 = torch.FloatTensor([0]).to(self.device)

        if mode != 'validation':
            mask_cells = batch_indices["cells"]
            mask_feats = batch_indices["feats"]
            
        # reconstruction loss, ||X - scale * C1 @ A_assos @ C2^t - b1 - b2^t||^2    
        for batch in range(self.nbatches):
            for idx_mod, mod in enumerate(self.mods): # use mods instead of self.mods to reduce computation
                if self.Xs[mod][batch] is not None:
                    scale = self.scales[mod][batch]
                    if mode == "validation":
                        A_assos = scale * (self.A_assos["shared"] + self.A_assos[mod + "_" + str(batch)])
                        loss1 += self.recon_loss(self.Xs[mod][batch], self.softmax(self.C_cells[str(batch)]), self.softmax(self.C_feats[mod]), A_assos, self.b_cells[mod][batch], self.b_feats[mod][batch])
                        # print("loss1_sub: {:.4e}".format(self.recon_loss(self.Xs[mod][batch], self.softmax(self.C_cells[str(batch)]), self.softmax(self.C_feats[mod]), A_assos, self.b_cells[mod][batch], self.b_feats[mod][batch]).item()) )
                    elif (mode == "C_cells") or (mode == "A_assos") or (mode[:7] == "C_feats" and mode[8:] == mod):
                        batch_X = self.Xs[mod][batch][np.ix_(mask_cells[batch], mask_feats[idx_mod])]
                        batch_C_cells = self.C_cells[str(batch)][mask_cells[batch],:]
                        batch_C_feats = self.C_feats[mod][mask_feats[idx_mod], :]
                        batch_A_asso = scale * (self.A_assos[mod + "_" + str(batch)] + self.A_assos["shared"])
                        batch_b_cells = self.b_cells[mod][batch][mask_cells[batch],:]
                        batch_b_feats = self.b_feats[mod][batch][:, mask_feats[idx_mod]]
                        # check if relu can actually be used, if initialize A_asso to be negative, then A_asso will never be updated as positive
                        loss1 += self.recon_loss(batch_X, self.softmax(batch_C_cells), self.softmax(batch_C_feats), batch_A_asso, batch_b_cells, batch_b_feats)
                        del batch_X, batch_C_cells, batch_C_feats, batch_A_asso, batch_b_cells, batch_b_feats       


        # association loss, calculate when mode is "validation" or "A_assos"
        if (mode == "validation") or (mode == "A_assos"):
            for batch in range(self.nbatches):
                for idx_mod, mod in enumerate(self.mods):
                    # make sure the l2 norm is minimized
                    if self.Xs[mod][batch] is not None:
                        loss2 += self.A_assos[mod + "_" + str(batch)].pow(2).sum()              
      
        loss = alpha[0] * loss1 + alpha[1] * loss2 

        return loss, loss1, loss2
    

    def train_func(self, T):
        best_loss = 1e12
        count = 0
        losses = []

        for t in range(T):
            mask_cells, mask_feats = self.sample_mini_batch()
            
            # update C_cells
            # print("update C_cells...")
            self.A_assos["shared"].requires_grad = False
            for i, mod in enumerate(self.mods):
                self.C_feats[mod].requires_grad = False
            for idx in self.A_assos.keys():
                self.A_assos[idx].requires_grad = False
                        
            for batch in range(self.nbatches):
                self.C_cells[str(batch)].requires_grad = True

            # sanity check
            for idx in self.C_cells.keys():
                assert self.C_cells[idx].requires_grad
            for idx in self.C_feats.keys():
                assert not self.C_feats[idx].requires_grad
            for idx in self.A_assos.keys():
                assert not self.A_assos[idx].requires_grad

            # update gradient    
            loss, *_ = self.batch_loss(mode = "C_cells", alpha = self.alpha, batch_indices = {"cells": mask_cells, "feats": mask_feats})
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # update C_feats
            # print("update C_feats...")
            for batch in range(self.nbatches):
                self.C_cells[str(batch)].requires_grad = False
            for i, mod in enumerate(self.mods):
                self.C_feats[mod].requires_grad = True

                # sanity check
                for idx in self.C_cells.keys():
                    assert not self.C_cells[idx].requires_grad
                for idx in self.C_feats.keys():
                    if idx != mod:
                        assert not self.C_feats[idx].requires_grad
                    elif idx == mod:
                        assert self.C_feats[idx].requires_grad
                for idx in self.A_assos.keys():
                    assert not self.A_assos[idx].requires_grad

                # update gradient  
                loss, *_ = self.batch_loss(mode = "C_feats_" + mod, alpha = self.alpha, batch_indices = {"cells": mask_cells, "feats": mask_feats})
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # update only one C_feats a time
                self.C_feats[mod].requires_grad = False

            # update A_assos:
            # print("update A_assos...")
            for idx in self.A_assos.keys():
                self.A_assos[idx].requires_grad = True

            # sanity check
            for idx in self.C_cells.keys():
                assert not self.C_cells[idx].requires_grad
            for idx in self.C_feats.keys():
                assert not self.C_feats[idx].requires_grad
            for idx in self.A_assos.keys():
                assert self.A_assos[idx].requires_grad

            loss, *_ = self.batch_loss(mode = "A_assos", alpha = self.alpha, batch_indices = {"cells": mask_cells, "feats": mask_feats})                    
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # no non-negative constraint
            with torch.no_grad():
                self.A_assos["shared"].data = self.A_assos["shared"] * (self.A_assos["shared"] > 0)
            
            
            with torch.no_grad():
                for batch in range(self.nbatches):
                    for idx_mod, mod in enumerate(self.mods):
                        if self.Xs[mod][batch] is not None:
                            batch_X = self.Xs[mod][batch][np.ix_(mask_cells[batch], mask_feats[idx_mod])]
                            batch_C_cells = self.C_cells[str(batch)][mask_cells[batch],:]
                            batch_C_feats = self.C_feats[mod][mask_feats[idx_mod], :]
                            batch_A_asso = self.A_assos[mod + "_" + str(batch)] + self.A_assos["shared"]
                            batch_b_feats = self.b_feats[mod][batch][:, mask_feats[idx_mod]]
                            batch_b_cells = self.b_cells[mod][batch][mask_cells[batch],:]
                            # update scale term:
                            scale = torch.trace((batch_X - batch_b_cells - batch_b_feats).t() @ (self.softmax(batch_C_cells) @ batch_A_asso @ self.softmax(batch_C_feats).t()))
                            scale = scale/(torch.trace((self.softmax(batch_C_cells) @ batch_A_asso @ self.softmax(batch_C_feats).t()).t() @ (self.softmax(batch_C_cells) @ batch_A_asso @ self.softmax(batch_C_feats).t())))
                            self.scales[mod][batch] = scale
                            # update bias term:
                            batch_b_feats = self.b_feats[mod][batch][:, mask_feats[idx_mod]]
                            self.b_cells[mod][batch][mask_cells[batch], :] = torch.mean(batch_X - self.scales[mod][batch] * self.softmax(batch_C_cells) @ batch_A_asso @ self.softmax(batch_C_feats).t() - batch_b_feats, dim = 1)[:,None]
                            batch_b_cells = self.b_cells[mod][batch][mask_cells[batch],:]
                            self.b_feats[mod][batch][:, mask_feats[idx_mod]] = torch.mean(batch_X - self.scales[mod][batch] * self.softmax(batch_C_cells) @ batch_A_asso @ self.softmax(batch_C_feats).t() - batch_b_cells, dim = 0)[None,:]


            
            # validation       
            if ((t+1) % self.interval == 0) | (t == 0):
                with torch.no_grad():
                    loss, loss1, loss2 = self.batch_loss(mode = "validation", alpha = self.alpha)

                    # print(self.A_assos["shared"])
                    # print(self.A_assos["rna_0"])
                    # print(self.A_assos["rna_1"])
                
                print('Epoch {}, Validating Loss: {:.4f}'.format(t + 1, loss.item()))
                info = [
                    'loss reconstruction: {:.5f}'.format(loss1.item()),
                    'loss regularization: {:.5f}'.format(loss2.item())
                ]
                for i in info:
                    print("\t", i)
                
                losses.append(loss.item())
                
                '''
                # update for early stopping 
                if loss.item() < best_loss:# - 0.01 * abs(best_loss):
                    
                    best_loss = loss.item()
                    # should save the whole model instead of just the state dict
                    # torch.save(self.state_dict(), f'../check_points/real_{self.N_cell}.pt')
                    count = 0
                else:
                    count += 1
                    print(count)
                    if count % int(T/self.interval) == 0:
                        self.optimizer.param_groups[0]['lr'] *= 0.5
                        print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < 1e-6:
                            break
                        else:
                            self.load_state_dict(torch.load(f'../check_points/real_{self.N_cell}.pt'))
                            count = 0                            
                '''
        return losses                            


class scmomat_retrain(Module):
    def __init__(self, model, counts, labels, lamb = 0.01, lr = 1e-2, seed = 0, device = device):
        """\
        Description:
        ------------
            Retraining the feature factors and association using the cell cluster labels
        Parameters:
        ------------
            model: the old cfrm model
            labels: the cell cluster labels, should be a list of arrays, each element within the list is the cluster assignment of cells within the corresponding batch
            lr: learning rate
            seed: seed
        """
        super().__init__()
        # construct cell factors according to cell cluster labels
        n_clusts = np.max(np.concatenate(labels, axis = 0)) + 1
        C_cells = {}
        for batch in range(model.nbatches):
            C_cells[str(batch)] = np.zeros((model.C_cells[str(batch)].shape[0], n_clusts))
            for clust in range(n_clusts):
                C_cells[str(batch)][labels[batch] == clust, clust] = 100     
        
        # initialize parameters
        self.nbatches = model.nbatches
        # make sure that the input count matrix have the same number of batches
        assert self.nbatches == counts["nbatches"]
        self.N_cell = C_cells["0"].shape[1]
        self.N_feat = C_cells["0"].shape[1]
        self.batch_size = model.batch_size
        self.interval = model.interval
        self.alpha = [1000, 1000 * lamb]
        self.device = device
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # instead of directly using model.mods, use mode from counts in case new modes are added.
        self.mods = [mod for mod in counts.keys() if (mod != "feats_name") & (mod != "nbatches")]
        self.Xs = {}
        # mod include: RNA, ATAC, PROTEIN, etc.
        for mod in self.mods:
            self.Xs[mod] = []
            for counts_mod in counts[mod]:
                if counts_mod is not None:
                    self.Xs[mod].append(torch.FloatTensor(counts_mod).to(self.device))
                else:
                    self.Xs[mod].append(None)

        # name of the features
        if "feats_name" in counts.keys():
            self.feats_name = counts["feats_name"]
        else:
            self.feats_name = None
        
        # self.C_cells should not be updated
        self.C_cells = {}
        self.C_feats = ParameterDict({})
        self.A_assos = ParameterDict({})
        self.b_cells = {}
        self.b_feats = {}
        self.scales = {}

        for batch in range(self.nbatches):
            self.C_cells[str(batch)] = torch.FloatTensor(C_cells[str(batch)]).to(self.device)

        # create C_feats, matrices exists for all mods
        for mod in self.mods:
            for batch in range(self.nbatches):
                if self.Xs[mod][batch] is not None:
                    self.C_feats[mod] = Parameter(torch.rand((self.Xs[mod][batch].shape[1], self.N_feat), device = self.device))
                    break

        self.A_assos["shared"] = Parameter(torch.rand((self.N_cell, self.N_feat), device = self.device))
        for mod in self.mods:
            for batch in range(self.nbatches):
                if self.Xs[mod][batch] is not None:
                    self.A_assos[mod + "_" + str(batch)] = Parameter(torch.rand((self.N_cell, self.N_feat), device = self.device))
       
        for mod in self.mods:
            self.b_cells[mod] = {}
            self.b_feats[mod] = {}
            for batch in range(self.nbatches):
                if self.Xs[mod][batch] is not None:
                    self.b_cells[mod][batch] = torch.zeros(self.Xs[mod][batch].shape[0], 1).to(self.device)
                    self.b_feats[mod][batch] = torch.zeros(1, self.Xs[mod][batch].shape[1]).to(self.device)

                self.scales[mod] = torch.FloatTensor([1] * self.nbatches).to(self.device)

        
        self.optimizer = opt.Adam(self.parameters(), lr=lr)

    @staticmethod
    def softmax(X: Tensor):
        return torch.softmax(X, dim = 1)
    
    @staticmethod
    def recon_loss(X, C1, C2, Sigma, b1, b2):
        return (X - C1 @ Sigma @ C2.t() - b1 - b2).pow(2).mean()

    @staticmethod
    def cosine_loss(A, B):
        return -torch.trace(A.t() @ B)/(torch.norm(A) + 1e-6)/(torch.norm(B) + 1e-6)    

    def sample_mini_batch(self):
        """\
        Sample mini batch
        """
        mask_cells = []
        mask_feats = []
        # sample mini_batch for each cell dimension
        for batch in range(self.nbatches):
            mask_cells.append(np.random.choice(self.C_cells[str(batch)].shape[0], int(self.C_cells[str(batch)].shape[0] * self.batch_size), replace=False))
        
        # sample mini_batch for each feature dimension
        for mod in self.mods:
            mask_feats.append(np.random.choice(self.C_feats[mod].shape[0], int(self.C_feats[mod].shape[0] * self.batch_size), replace=False))
        
        return mask_cells, mask_feats

    def batch_loss(self, mode, alpha, batch_indices = None):
        """\
            Calculate overall loss term
        """
        # init
        loss1 = torch.FloatTensor([0]).to(self.device)
        loss2 = torch.FloatTensor([0]).to(self.device)

        if mode != 'validation':
            mask_cells = batch_indices["cells"]
            mask_feats = batch_indices["feats"]
            
        # reconstruction loss, ||X - scale * C1 @ A_assos @ C2^t - b1 - b2^t||^2    
        for batch in range(self.nbatches):
            for idx_mod, mod in enumerate(self.mods): # use mods instead of self.mods to reduce computation
                if self.Xs[mod][batch] is not None:
                    scale = self.scales[mod][batch]
                    if mode == "validation":
                        A_assos = scale * (self.A_assos["shared"] + self.A_assos[mod + "_" + str(batch)])
                        loss1 += self.recon_loss(self.Xs[mod][batch], self.softmax(self.C_cells[str(batch)]), self.softmax(self.C_feats[mod]), A_assos, self.b_cells[mod][batch], self.b_feats[mod][batch])
                        # print("loss1_sub: {:.4e}".format(self.recon_loss(self.Xs[mod][batch], self.softmax(self.C_cells[str(batch)]), self.softmax(self.C_feats[mod]), A_assos, self.b_cells[mod][batch], self.b_feats[mod][batch]).item()) )
                    elif (mode == "C_cells") or (mode == "A_assos") or (mode[:7] == "C_feats" and mode[8:] == mod):
                        batch_X = self.Xs[mod][batch][np.ix_(mask_cells[batch], mask_feats[idx_mod])]
                        batch_C_cells = self.C_cells[str(batch)][mask_cells[batch],:]
                        batch_C_feats = self.C_feats[mod][mask_feats[idx_mod], :]
                        batch_A_asso = scale * (self.A_assos[mod + "_" + str(batch)] + self.A_assos["shared"])
                        batch_b_cells = self.b_cells[mod][batch][mask_cells[batch],:]
                        batch_b_feats = self.b_feats[mod][batch][:, mask_feats[idx_mod]]
                        # check if relu can actually be used, if initialize A_asso to be negative, then A_asso will never be updated as positive
                        loss1 += self.recon_loss(batch_X, self.softmax(batch_C_cells), self.softmax(batch_C_feats), batch_A_asso, batch_b_cells, batch_b_feats)
                        del batch_X, batch_C_cells, batch_C_feats, batch_A_asso, batch_b_cells, batch_b_feats       


        # association loss, calculate when mode is "validation" or "A_assos"
        if (mode == "validation") or (mode == "A_assos"):
            for batch in range(self.nbatches):
                for idx_mod, mod in enumerate(self.mods):
                    # make sure the l2 norm is minimized
                    if self.Xs[mod][batch] is not None:
                        loss2 += self.A_assos[mod + "_" + str(batch)].pow(2).sum()              
      
        
        loss = alpha[0] * loss1 + alpha[1] * loss2

        return loss, loss1, loss2

    def train(self, T):
        """\
        Training function, T is the number of iterations
        """
        best_loss = 1e12
        count = 0
        losses = []

        for t in range(T):
            mask_cells, mask_feats = self.sample_mini_batch()

            # update C_feats
            for i, mod in enumerate(self.mods):
                self.C_feats[mod].requires_grad = True
                loss, *_ = self.batch_loss(mode = "C_feats_" + mod, alpha = self.alpha, batch_indices = {"cells": mask_cells, "feats": mask_feats})
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # update only one C_feats a time
                self.C_feats[mod].requires_grad = False

            # update A_assos:
            # self.A_assos["shared"].requires_grad = True
            for idx in self.A_assos.keys():
                self.A_assos[idx].requires_grad = True
            loss, *_ = self.batch_loss(mode = "A_assos", alpha = self.alpha, batch_indices = {"cells": mask_cells, "feats": mask_feats})                    
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # no non-negative constraint
            with torch.no_grad():
                self.A_assos["shared"].data = self.A_assos["shared"] * (self.A_assos["shared"] > 0)

            # update bias term
            with torch.no_grad():
                for batch in range(self.nbatches):
                    for idx_mod, mod in enumerate(self.mods):
                        if self.Xs[mod][batch] is not None:
                            batch_X = self.Xs[mod][batch][np.ix_(mask_cells[batch], mask_feats[idx_mod])]
                            batch_C_cells = self.C_cells[str(batch)][mask_cells[batch],:]
                            batch_C_feats = self.C_feats[mod][mask_feats[idx_mod], :]
                            batch_A_asso = self.A_assos[mod + "_" + str(batch)] + self.A_assos["shared"]
                            batch_b_feats = self.b_feats[mod][batch][:, mask_feats[idx_mod]]
                            batch_b_cells = self.b_cells[mod][batch][mask_cells[batch],:]
                            # update scale term:
                            scale = torch.trace((batch_X - batch_b_cells - batch_b_feats).t() @ (self.softmax(batch_C_cells) @ batch_A_asso @ self.softmax(batch_C_feats).t()))
                            scale = scale/(torch.trace((self.softmax(batch_C_cells) @ batch_A_asso @ self.softmax(batch_C_feats).t()).t() @ (self.softmax(batch_C_cells) @ batch_A_asso @ self.softmax(batch_C_feats).t())))
                            self.scales[mod][batch] = scale
                            # update bias term:
                            batch_b_feats = self.b_feats[mod][batch][:, mask_feats[idx_mod]]
                            self.b_cells[mod][batch][mask_cells[batch], :] = torch.mean(batch_X - scale * self.softmax(batch_C_cells) @ batch_A_asso @ self.softmax(batch_C_feats).t() - batch_b_feats, dim = 1)[:,None]
                            batch_b_cells = self.b_cells[mod][batch][mask_cells[batch],:]
                            self.b_feats[mod][batch][:, mask_feats[idx_mod]] = torch.mean(batch_X - scale * self.softmax(batch_C_cells) @ batch_A_asso @ self.softmax(batch_C_feats).t() - batch_b_cells, dim = 0)[None,:]
            
            # validation       
            if ((t+1) % self.interval == 0) | (t == 0):
                with torch.no_grad():
                    loss, loss1, loss2 = self.batch_loss(mode = "validation", alpha = self.alpha)
                
                print('Epoch {}, Validating Loss: {:.4f}'.format(t + 1, loss.item()))
                info = [
                    'loss reconstruction: {:.5f}'.format(loss1.item()),
                    'loss regularization: {:.5f}'.format(loss2.item())
                ]
                for i in info:
                    print("\t", i)
                
                losses.append(loss.item())
                
                # # update for early stopping 
                # if loss.item() < best_loss:# - 0.01 * abs(best_loss):
                    
                #     best_loss = loss.item()
                #     torch.save(self.state_dict(), f'../check_points/real_{self.N_cell}.pt')
                #     count = 0
                # else:
                #     count += 1
                #     print(count)
                #     if count % int(T/self.interval) == 0:
                #         self.optimizer.param_groups[0]['lr'] *= 0.5
                #         print('Epoch: {}, shrink lr to {:.4f}'.format(t + 1, self.optimizer.param_groups[0]['lr']))
                #         if self.optimizer.param_groups[0]['lr'] < 1e-6:
                #             break
                #         else:
                #             self.load_state_dict(torch.load(f'../check_points/real_{self.N_cell}.pt'))
                #             count = 0                            
                        
         
        return losses        

