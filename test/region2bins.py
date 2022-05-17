# In[]
import numpy as np
import pandas as pd 
import sys, os 
from scipy.sparse import load_npz, save_npz, csr_matrix
from scipy.io import mmwrite, mmread
import warnings
warnings.filterwarnings("ignore")

# In[]
data_dir = "../data/real/MOp/"
data_dir = "../data/real/ASAP-PBMC/"
data_dir = "../data/real/diag/mouse_brain_cortex/"
data_dir = "../data/real/diag/healthy_hema/topgenes_2000/BMMC/"
data_dir = "../data/real/diag/Xichen/"
data_dir = "../data/real/MOp_ext/"
data_dir = "../data/real/diag/Xichen/remove_celltype/"

bin_size = 100000

# read in regions
regions = pd.read_csv(data_dir + "regions.txt", header = None).values.squeeze()
# store regions into dataframe
region_ranges = pd.DataFrame(columns = ["chr", "start", "end"])
for idx, region in enumerate(regions):
    region_ranges = region_ranges.append({"chr": region.split("_")[0], "start": eval(region.split("_")[1]), "end": eval(region.split("_")[2])}, ignore_index = True)
region_ranges.index = regions

# create bins
chroms = np.sort(np.unique(region_ranges["chr"].values.squeeze()))
bins = []
for chrom in chroms:
    regions_chrom = region_ranges[region_ranges["chr"] == chrom]
    # the samllest and largest chromatin regions in the current chromosome
    end_idx = np.max(regions_chrom["end"].values)
    start_idx = np.min(regions_chrom["start"].values)
    # number of bins for current chromatin
    nbins = (end_idx - start_idx)//bin_size + 1
    bin_bps = np.arange(start_idx, end_idx, bin_size)
    bins.extend([chrom + "_" + str(x) + "_" + str(x + bin_size) for x in bin_bps])

# create a region by bin matrix
region2bin = pd.DataFrame(data = 0, index = regions, columns = bins)
for chrom in chroms:
    # regions in current chormatin
    regions_chrom = region_ranges[region_ranges["chr"] == chrom]
    # the samllest and largest chromatin regions in the current chromosome
    end_idx = np.max(regions_chrom["end"].values)
    start_idx = np.min(regions_chrom["start"].values)
    # number of bins for current chromatin
    nbins = (end_idx - start_idx)//bin_size + 1
    # segementing points along the chromatin
    bin_bps = np.arange(start_idx, end_idx, bin_size)
    # assign regions to bins
    for region in regions_chrom.index.values:
        bin = chrom + "_" + str(np.max(bin_bps[bin_bps <= regions_chrom.loc[region, "start"]])) + "_" + str(np.max(bin_bps[bin_bps <=  regions_chrom.loc[region, "start"]]) + bin_size)
        region2bin.loc[region, bin] += 1

# remove zero counts bins
nonzero_bins = np.where(np.sum(region2bin.values, axis = 0) != 0)[0]
region2bin = region2bin.iloc[:, nonzero_bins]
np.savetxt(data_dir + "bins.txt", region2bin.columns.values, fmt = "%s")

# In[] Generate bin by cell matrix

n_batches = 3
for batch in range(n_batches):
    try:
        counts_atac = load_npz(os.path.join(data_dir, 'RxC' + str(batch + 1) + ".npz"))
        counts_bin = csr_matrix(region2bin.T.values) * counts_atac
        mmwrite(data_dir + "BxC" + str(batch + 1) + ".mtx", csr_matrix(counts_bin))
    except:
        counts_atac = None






# %%
