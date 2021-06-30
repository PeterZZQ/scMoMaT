# In[0]
from sklearn.metrics import adjusted_rand_score
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys, os

# In[1]
runs = 5

dir1 = '../data/simulated/'
dir2 = "./results_multi_liger_quantile/multi3/"
paths = ['2b5c_sigma0.1_b2_1/', '2b5c_sigma0.1_b2_2/', '2b5c_sigma0.2_b2_1/', '2b5c_sigma0.2_b2_2/', '2b5c_sigma0.3_b2_1/', '2b5c_sigma0.3_b2_2/', \
    '2b4c_sigma0.1_b2_1/', '2b4c_sigma0.1_b2_2/', '2b4c_sigma0.2_b2_1/', '2b4c_sigma0.2_b2_2/', '2b4c_sigma0.3_b2_1/', '2b4c_sigma0.3_b2_2/', \
        '2b3c_sigma0.1_b2_1/', '2b3c_sigma0.1_b2_2/', '2b3c_sigma0.2_b2_1/', '2b3c_sigma0.2_b2_2/', '2b3c_sigma0.3_b2_1/', '2b3c_sigma0.3_b2_2/',\
            '2b5c_sigma0.1_b1_1/', '2b5c_sigma0.1_b1_2/', '2b5c_sigma0.2_b1_1/', '2b5c_sigma0.2_b1_2/', '2b5c_sigma0.3_b1_1/', '2b5c_sigma0.3_b1_2/', \
                '2b4c_sigma0.1_b1_1/', '2b4c_sigma0.1_b1_2/', '2b4c_sigma0.2_b1_1/', '2b4c_sigma0.2_b1_2/', '2b4c_sigma0.3_b1_1/', '2b4c_sigma0.3_b1_2/', \
                    '2b3c_sigma0.1_b1_1/', '2b3c_sigma0.1_b1_2/', '2b3c_sigma0.2_b1_1/', '2b3c_sigma0.2_b1_2/', '2b3c_sigma0.3_b1_1/', '2b3c_sigma0.3_b1_2/',\
                        '2b3c_sigma0.4_b1_1/', '2b3c_sigma0.4_b1_2/', '2b4c_sigma0.4_b1_1/', '2b4c_sigma0.4_b1_2/', '2b5c_sigma0.4_b1_1/', '2b5c_sigma0.4_b1_2/',\
                            '2b3c_sigma0.5_b1_1/', '2b3c_sigma0.5_b1_2/', '2b4c_sigma0.5_b1_1/', '2b4c_sigma0.5_b1_2/', '2b5c_sigma0.5_b1_1/', '2b5c_sigma0.5_b1_2/']

for path in paths:
    print(path[:-1])
    label_b1 = pd.read_csv(os.path.join(dir1 + path, "cell_label1.txt"), index_col=0, sep = "\t")["pop"].values.squeeze()
    label_b2 = pd.read_csv(os.path.join(dir1 + path, "cell_label2.txt"), index_col=0, sep = "\t")["pop"].values.squeeze()
    pred_label = [x for x in pd.read_csv(os.path.join(dir2 + path[:-1] + "_clust_id.csv"), index_col=0, sep = ",")["x"].values]
    pred_label = np.array(pred_label)
    ari = [adjusted_rand_score(labels_pred = pred_label, labels_true = np.concatenate((label_b1, label_b2), axis = 0))] * runs 
    ari = np.array(ari)
    np.save(file = dir2 + path[:-1] + "_ari_liger_quantile.npy", arr = ari)
# In[2]
dir2 = "./results_multi_liger_quantile/multi4/"
for path in paths:
    label_b1 = pd.read_csv(os.path.join(dir1 + path, "cell_label1.txt"), index_col=0, sep = "\t")["pop"].values.squeeze()
    label_b2 = pd.read_csv(os.path.join(dir1 + path, "cell_label2.txt"), index_col=0, sep = "\t")["pop"].values.squeeze()
    pred_label = [x for x in pd.read_csv(os.path.join(dir2 + path[:-1] + "_clust_id.csv"), index_col=0, sep = ",")["x"].values]
    pred_label = np.array(pred_label)
    ari = [adjusted_rand_score(labels_pred = pred_label, labels_true = np.concatenate((label_b1, label_b2), axis = 0))] * runs 
    ari = np.array(ari)
    np.save(file = dir2 + path[:-1] + "_ari_liger_quantile.npy", arr = ari)

# In[3]

runs = 5

dir1 = '../data/simulated/'
dir2 = "./results_multi_liger/multi3/"
paths = ['2b5c_sigma0.1_b2_1/', '2b5c_sigma0.1_b2_2/', '2b5c_sigma0.2_b2_1/', '2b5c_sigma0.2_b2_2/', '2b5c_sigma0.3_b2_1/', '2b5c_sigma0.3_b2_2/', \
    '2b4c_sigma0.1_b2_1/', '2b4c_sigma0.1_b2_2/', '2b4c_sigma0.2_b2_1/', '2b4c_sigma0.2_b2_2/', '2b4c_sigma0.3_b2_1/', '2b4c_sigma0.3_b2_2/', \
        '2b3c_sigma0.1_b2_1/', '2b3c_sigma0.1_b2_2/', '2b3c_sigma0.2_b2_1/', '2b3c_sigma0.2_b2_2/', '2b3c_sigma0.3_b2_1/', '2b3c_sigma0.3_b2_2/',\
            '2b5c_sigma0.1_b1_1/', '2b5c_sigma0.1_b1_2/', '2b5c_sigma0.2_b1_1/', '2b5c_sigma0.2_b1_2/', '2b5c_sigma0.3_b1_1/', '2b5c_sigma0.3_b1_2/', \
                '2b4c_sigma0.1_b1_1/', '2b4c_sigma0.1_b1_2/', '2b4c_sigma0.2_b1_1/', '2b4c_sigma0.2_b1_2/', '2b4c_sigma0.3_b1_1/', '2b4c_sigma0.3_b1_2/', \
                    '2b3c_sigma0.1_b1_1/', '2b3c_sigma0.1_b1_2/', '2b3c_sigma0.2_b1_1/', '2b3c_sigma0.2_b1_2/', '2b3c_sigma0.3_b1_1/', '2b3c_sigma0.3_b1_2/',\
                        '2b3c_sigma0.4_b1_1/', '2b3c_sigma0.4_b1_2/', '2b4c_sigma0.4_b1_1/', '2b4c_sigma0.4_b1_2/', '2b5c_sigma0.4_b1_1/', '2b5c_sigma0.4_b1_2/',\
                            '2b3c_sigma0.5_b1_1/', '2b3c_sigma0.5_b1_2/', '2b4c_sigma0.5_b1_1/', '2b4c_sigma0.5_b1_2/', '2b5c_sigma0.5_b1_1/', '2b5c_sigma0.5_b1_2/']

for path in paths:
    print(path[:-1])
    label_b1 = pd.read_csv(os.path.join(dir1 + path, "cell_label1.txt"), index_col=0, sep = "\t")["pop"].values.squeeze()
    label_b2 = pd.read_csv(os.path.join(dir1 + path, "cell_label2.txt"), index_col=0, sep = "\t")["pop"].values.squeeze()
    pred_label = [x for x in pd.read_csv(os.path.join(dir2 + path[:-1] + "_clust_id.csv"), index_col=0, sep = ",")["x"].values]
    pred_label = np.array(pred_label)
    ari = [adjusted_rand_score(labels_pred = pred_label, labels_true = np.concatenate((label_b1, label_b2), axis = 0))] * runs 
    ari = np.array(ari)
    np.save(file = dir2 + path[:-1] + "_ari_liger.npy", arr = ari)

# In[4]
dir2 = "./results_multi_liger/multi4/"
for path in paths:
    label_b1 = pd.read_csv(os.path.join(dir1 + path, "cell_label1.txt"), index_col=0, sep = "\t")["pop"].values.squeeze()
    label_b2 = pd.read_csv(os.path.join(dir1 + path, "cell_label2.txt"), index_col=0, sep = "\t")["pop"].values.squeeze()
    pred_label = [x for x in pd.read_csv(os.path.join(dir2 + path[:-1] + "_clust_id.csv"), index_col=0, sep = ",")["x"].values]
    pred_label = np.array(pred_label)
    ari = [adjusted_rand_score(labels_pred = pred_label, labels_true = np.concatenate((label_b1, label_b2), axis = 0))] * runs 
    ari = np.array(ari)
    np.save(file = dir2 + path[:-1] + "_ari_liger.npy", arr = ari)


# %%
