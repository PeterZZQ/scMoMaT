import pandas as pd 
import numpy as np 
from scipy.sparse import load_npz, csr_matrix

from matplotlib import pyplot as plt

from sklearn.datasets import make_checkerboard
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score

gxr = load_npz("BMMC/GxR.npz").todense().astype(np.float)

for n_clusters in range(3,10):
    fig = plt.figure(figsize=(30,10))
    axs = fig.subplots(1,2)
    axs[0].matshow(gxr, cmap=plt.cm.Blues)
    axs[0].set_title("Shuffled dataset")
    # too sparse, the method will not work
    model = SpectralBiclustering(n_clusters=(n_clusters, n_clusters), method='bistochastic', random_state=0)
    model.fit(gxr)
    # rows and columns, like ground truth
    # score = consensus_score(model.biclusters_, (rows[:, row_idx], columns[:, col_idx]))
    # print("consensus score: {:.1f}".format(score))

    fit_data = gxr[np.argsort(model.row_labels_)]
    fit_data = gxr[:, np.argsort(model.column_labels_)]

    axs[1].matshow(fit_data, cmap=plt.cm.Blues)
    axs[1].set_title("After biclustering; rearranged to show biclusters")

    axs[2].matshow(np.outer(np.sort(model.row_labels_) + 1,
                        np.sort(model.column_labels_) + 1),
                cmap=plt.cm.Blues)
    plt.title("Checkerboard structure of rearranged data")
    
    fig.savefig("cluster_" + str(n_clusters) + ".png")
    plt.show()