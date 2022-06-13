# In[]
import numpy as np 
import pandas as pd 
from scipy.io import mmread, mmwrite 
import sys, os
import scanpy as sc
from anndata import AnnData
from scipy.sparse import save_npz, csr_matrix, load_npz
import warnings 
# warnings.filterwarnings("ignore")

# In[] Health hema
path = "../../data/real/diag/healthy_hema/"
counts_rna = mmread(path + "multimap/GxC2_raw.mtx").todense()
meta_rna = pd.read_csv(path + "multimap/meta_c2.csv", index_col = 0, sep = "\t")
genes = pd.read_csv(path + "multimap/genes.csv", header = None)
counts_rna = pd.DataFrame(data = counts_rna, index = genes.values.squeeze(), columns = meta_rna.index.values.squeeze())

counts_protein = mmread(path + "multimap/PxC2_sub_raw.mtx").todense()
meta_protein = pd.read_csv(path + "multimap/meta_c2_sub.csv", index_col = 0, sep = "\t")
protein = pd.read_csv(path + "multimap/proteins.csv", header = None)
counts_protein = pd.DataFrame(data = counts_protein, index = protein.values.squeeze(), columns = meta_protein.index.values.squeeze())

counts_atac = mmread(path + "multimap/RxC1_raw.mtx")
meta_atac = pd.read_csv(path + "multimap/meta_c1.csv", index_col = 0, sep = "\t")
regions_full = pd.read_csv(path + "multimap/regions.csv", header = None).values.squeeze()
counts_atac = pd.DataFrame(data = counts_atac.todense(), index = regions_full, columns = meta_atac.index.values.squeeze())

gact = pd.read_csv(path + "multimap/seurat_upstream_2000.csv")
counts_atac_tmp = counts_atac.loc[gact["peak"].values.squeeze(), :]
counts_atac_tmp.index = gact["gene.name"].values.squeeze()
counts_atac2rna= counts_atac_tmp.groupby(level = 0).sum()

# BMMC
result_paths = path + "multimap/BMMC/"
meta_atac_bm = pd.read_csv(path + "topgenes_1000/BMMC/meta_c1.csv", index_col = 0, sep = ",")
meta_rna_bm = pd.read_csv(path + "topgenes_1000/BMMC/meta_c2.csv", index_col = 0, sep = ",")
counts_rna_bm = counts_rna.loc[:, meta_rna_bm.index.values]
counts_protein_bm = counts_protein.loc[:, meta_rna_bm.index.values]
counts_atac_bm = counts_atac.loc[:, meta_atac_bm.index.values]
counts_atac2rna_bm = counts_atac2rna.loc[:, meta_atac_bm.index.values]

meta_rna_bm.to_csv(result_paths + "meta_c2.csv")
genes.to_csv(result_paths + "genes.csv", header = False, index = False)
save_npz(result_paths + "GxC2_raw.npz", csr_matrix(counts_rna_bm.values))
mmwrite(result_paths + "GxC2_raw.mtx", csr_matrix(counts_rna_bm.values))
meta_atac_bm.to_csv(result_paths + "meta_c1.csv")
regions_full.to_csv(result_paths + "regions.csv", header = False, index = False)
save_npz(result_paths + "RxC1_raw.npz", csr_matrix(counts_atac_bm.values))
mmwrite(result_paths + "RxC1_raw.mtx", csr_matrix(counts_atac_bm.values))

rna = AnnData(X = csr_matrix(counts_rna_bm.T.values))
rna.var.index = counts_rna_bm.index.values
rna.obs = meta_rna_bm
atac_peaks = AnnData(X = csr_matrix(counts_atac_bm.T.values))
atac_peaks.var.index = counts_atac_bm.index.values
atac_peaks.obs = meta_atac_bm
atac_genes = AnnData(X = csr_matrix(counts_atac2rna_bm.T.values))
atac_genes.var.index = counts_atac2rna_bm.index.values
atac_genes.obs = meta_atac_bm

# just conduct log transform, as is stated in multimap
sc.pp.log1p(rna)
sc.pp.log1p(atac_genes)
rna.obs["source"] = "C2"
atac_genes.obs["source"] = "C1"
atac_peaks.obs["source"] = "C1"

rna.write_h5ad(result_paths + "rna.h5ad")
atac_genes.write_h5ad(result_paths + "atac-genes.h5ad")
atac_peaks.write_h5ad(result_paths + "atac-peaks.h5ad")

# In[] Mouse brain cortex
path = "../../data/real/diag/mouse_brain_cortex/"
counts_rna = mmread(path + "multimap/counts_rna.mtx")
meta_rna = pd.read_csv(path + "multimap/meta_cells_rna.csv", index_col = 0, sep = "\t")
meta_gene = pd.read_csv(path + "multimap/meta_genes.csv", index_col = 0, sep = "\t")
counts_rna = pd.DataFrame(data = counts_rna.todense(), index = meta_gene.values.squeeze(), columns = meta_rna.index.values.squeeze())

counts_atac = mmread(path + "multimap/counts_atac.mtx")
meta_atac = pd.read_csv(path + "multimap/meta_cells_atac.csv", index_col = 0, sep = "\t")
meta_region = pd.read_csv(path + "multimap/meta_regions.csv", index_col = 0, sep = "\t")
counts_atac = counts_atac.todense()
counts_atac = (counts_atac > 0).astype(int)

counts_atac = pd.DataFrame(data = counts_atac, 
                           index = meta_region.index.values.squeeze(), 
                           columns = meta_atac.index.values.squeeze())

gact = pd.read_csv(path + "multimap/seurat_gact_up2000.csv")
counts_atac_tmp = counts_atac.loc[gact["peak"].values.squeeze(), :]
counts_atac_tmp.index = gact["gene.name"].values.squeeze()
counts_atac2rna= counts_atac_tmp.groupby(level = 0).sum()


meta_rna = pd.read_csv(path + "meta_c1.csv", index_col = 0, sep = ",")
meta_atac = pd.read_csv(path + "meta_c2.csv", index_col = 0, sep = ",")
counts_rna = counts_rna.loc[:, meta_rna.index.values]
counts_atac = counts_atac.loc[:, meta_atac.index.values]
counts_atac2rna = counts_atac2rna.loc[:, meta_atac.index.values]

rna = AnnData(X = csr_matrix(counts_rna.T.values))
rna.var.index = counts_rna.index.values
rna.obs = meta_rna
atac_peaks = AnnData(X = csr_matrix(counts_atac.T.values))
atac_peaks.var.index = counts_atac.index.values
atac_peaks.obs = meta_atac
atac_genes = AnnData(X = csr_matrix(counts_atac2rna.T.values))
atac_genes.var.index = counts_atac2rna.index.values
atac_genes.obs = meta_atac

# just conduct log transform, as is stated in multimap
sc.pp.log1p(rna)
sc.pp.log1p(atac_genes)
rna.obs["source"] = "C1"
atac_genes.obs["source"] = "C2"
atac_peaks.obs["source"] = "C2"

result_paths = path + "multimap/"
rna.write_h5ad(result_paths + "rna.h5ad")
atac_genes.write_h5ad(result_paths + "atac-genes.h5ad")
atac_peaks.write_h5ad(result_paths + "atac-peaks.h5ad")

# In[]
###########################################################################################################################################
#
# Remove some cell types
#
###########################################################################################################################################
# Remove B cells in Spleen dataset
data_dir = "../../data/real/diag/Xichen/multimap/"
remove_dir = '../../data/real/diag/Xichen/remove_celltype/'
result_paths = "./"
rna = sc.read(filename = data_dir + "rna.h5ad")
atac_peaks = sc.read(filename = data_dir + "atac-peaks.h5ad")
atac_genes = sc.read(filename = data_dir + "atac-genes.h5ad")

meta_rna = pd.read_csv(os.path.join(remove_dir, 'meta_c1.csv'), index_col=0)
rna = rna[meta_rna.index.values.squeeze(), :]
meta_atac = pd.read_csv(os.path.join(remove_dir, 'meta_c2.csv'), index_col=0)
atac_genes = atac_genes[meta_atac.index.values.squeeze(), :]
atac_peaks = atac_peaks[meta_atac.index.values.squeeze(), :]

rna.write_h5ad(result_paths + "rna.h5ad")
atac_genes.write_h5ad(result_paths + "atac-genes.h5ad")
atac_peaks.write_h5ad(result_paths + "atac-peaks.h5ad")

# In[] Simulated test scenario 1
np.random.seed(0)
path = "../../data/simulated/6b16c_test_10_large/"
path_new = path + "unequal2/"
if not os.path.exists(path_new):
    os.mkdir(path_new)

result_path = path_new + "multimap/"
if not os.path.exists(result_path):
    os.makedirs(result_path)
n_batches = 6
adata_rnas = []
adata_atacs = []
adata_atac2rnas = []
labels = []
# association matrix
A = np.loadtxt(os.path.join(path, 'region2gene.txt'), delimiter = "\t")
np.savetxt(path_new + 'region2gene.txt', A, delimiter = "\t")
for batch in range(n_batches):        
    label = pd.read_csv(os.path.join(path, 'cell_label' + str(batch + 1) + '.txt'), index_col=0, sep = "\t")["pop"].values.squeeze()
    labels.append(label)
    # read in atac-seq
    counts_atac = np.loadtxt(os.path.join(path, 'RxC' + str(batch + 1) + ".txt"), delimiter = "\t").T
    counts_atac = (counts_atac > 0).astype(np.float)
    barcodes = np.array(["batch_" + str(batch) + ":cell_" + str(x) for x in range(counts_atac.shape[0])])
    regions = np.array(["region_" + str(x) for x in range(counts_atac.shape[1])])
    adata_atac = AnnData(csr_matrix(counts_atac))
    adata_atac.var.index = regions
    adata_atac.obs.index = barcodes
    adata_atac.obs["source"] = "C" + str(batch + 1)
    adata_atac.obs["pop"] = label
    print("read atac for batch" + str(batch + 1))

    # remove some cell types
    clusters = np.sort(np.unique(label))
    missing_cluster = np.random.choice(clusters, 5, replace = False)
    print(missing_cluster)
    missing_barcodes = barcodes[np.array([x for x in range(label.shape[0]) if label[x] in missing_cluster])]
    missing_barcodes = np.random.choice(missing_barcodes, size = int(1.0 * missing_barcodes.shape[0]), replace = False)
    barcodes2 = np.array([x for x in barcodes if x not in missing_barcodes])
    print(barcodes.shape)
    print(barcodes2.shape)
    adata_atac = adata_atac[barcodes2, :]
    counts_atac = adata_atac.X.todense()
    np.savetxt(path_new + "RxC" + str(batch + 1) + ".txt", counts_atac.T, delimiter = "\t")
    print(np.sort(np.unique(adata_atac.obs["pop"].values)))
    adata_atac.obs.to_csv(path_new + "cell_label" + str(batch + 1) + ".txt", sep = "\t")

    # read in rna-seq
    counts_rna = np.loadtxt(os.path.join(path, 'GxC' + str(batch + 1) + ".txt"), delimiter = "\t").T
    barcodes = np.array(["batch_" + str(batch) + ":cell_" + str(x) for x in range(counts_rna.shape[0])])
    genes = np.array(["gene_" + str(x) for x in range(counts_rna.shape[1])])
    adata_rna = AnnData(csr_matrix(counts_rna))
    adata_rna.var.index = genes
    adata_rna.obs.index = barcodes
    adata_rna.obs["source"] = "C" + str(batch + 1)
    adata_rna.obs["pop"] = label
    print("read rna for batch" + str(batch + 1))
    
    # remove some cell types
    adata_rna = adata_rna[barcodes2, :]
    counts_rna = adata_rna.X.todense().T
    np.savetxt(path_new + "GxC" + str(batch + 1) + ".txt", counts_rna, delimiter="\t")    
    
    adata_atac2rna = AnnData(csr_matrix(counts_atac @ A))
    adata_atac2rna.var = adata_rna.var
    adata_atac2rna.obs = adata_atac.obs

    # remove genes and regions that have only zero counts
    # sc.pp.filter_genes(adata_rna, min_cells = 1)
    sc.pp.log1p(adata_rna)
    # sc.pp.filter_genes(adata_atac, min_cells = 1)
    # sc.pp.filter_genes(adata_atac2rna, min_cells = 1)
    sc.pp.log1p(adata_atac2rna)

    
    # preprocess the count matrix
    adata_rnas.append(adata_rna)
    adata_atacs.append(adata_atac)
    adata_atac2rnas.append(adata_atac2rna)

    adata_rna.write_h5ad(result_path + "rna_" + str(batch+1) + ".h5ad")
    adata_atac.write_h5ad(result_path + "atac_peaks_" + str(batch+1) + ".h5ad")
    adata_atac2rna.write_h5ad(result_path + "atac_genes_" + str(batch+1) + ".h5ad")


# %%

