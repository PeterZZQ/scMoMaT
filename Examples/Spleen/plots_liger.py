# In[0]
from operator import index
import sys, os
sys.path.append('../../')
sys.path.append('../../src/')

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from umap import UMAP
import utils

# In[1]
# read in latent factor of liger
clusts = [8, 20, 30]
for clust in clusts:
    path = "./Liger/k_" + str(clust) + "/"
    data_dir = "../../data/real/diag/Xichen/"
    H1 = pd.read_csv(path + "H1.csv", index_col = 0).values
    H2 = pd.read_csv(path + "H2.csv", index_col = 0)
    H1_norm = pd.read_csv(path + "H1_norm.csv", index_col = 0).values
    H2_norm = pd.read_csv(path + "H2_norm.csv", index_col = 0).values

    labels = []
    for batch in range(0,2):
        labels.append(pd.read_csv(os.path.join(data_dir, 'meta_c' + str(batch + 1) + '.csv'), index_col=0)["cell_type"].values.squeeze())

    umap_op = UMAP(n_components = 2)
    x_umaps = umap_op.fit_transform(np.concatenate([H1, H2], axis = 0))
    x_umaps = [x_umaps[:labels[0].shape[0], :], x_umaps[labels[0].shape[0]:, :]]
    utils.plot_latent_ext(x_umaps, annos = labels, mode = "separate", save = path + "separate.png", figsize = (10,15), axis_label = "Latent")

    x_umaps_norm = umap_op.fit_transform(np.concatenate([H1_norm, H2_norm], axis = 0))
    x_umaps_norm = [x_umaps_norm[:labels[0].shape[0], :], x_umaps_norm[labels[0].shape[0]:, :]]
    utils.plot_latent_ext(x_umaps_norm, annos = labels, mode = "separate", save = path + "separate_norm.png", figsize = (10,15), axis_label = "Latent")


# %%
