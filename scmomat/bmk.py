import numpy as np
import pandas as pd 
import os
import subprocess

import scipy.special
from scipy.sparse import csr_matrix
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import silhouette_samples, silhouette_score
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph, NearestNeighbors
import scipy.stats as stats

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#
# Conservation of biological variation in single-cell data
#
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
########################################################################################
#
# KNN purity (https://rdrr.io/github/YosefLab/SymSim/man/cal_KNNP.html)
#
########################################################################################
def knn_purity(X, label, k = 30):
    """\
    Description:
    ------------
        Calculate the knn purity score.
    Parameters:
    ------------
        X: the embedding to calculate knn purity score upon.
        label: the label of the embedding.
        k: number of neighbors.
    Return:
        knn purity score
    """
    neighbor = NearestNeighbors(n_neighbors = k)
    neighbor.fit(X)
    # get test connectivity result 0-1 adj_matrix, mode = 'connectivity' by default
    G = neighbor.kneighbors_graph(X).toarray()
    score = 0
    for i in range(G.shape[0]):
        score += (np.sum(label[G[i,:].astype(np.bool)] == label[i]) - 1)/(k-1)
    return score/G.shape[0]


########################################################################################
#
# LABEL Trasfer accuracy from SCOT(https://github.com/rsinghlab/SCOT/blob/master/src/evals.py)
#
########################################################################################
def transfer_accuracy(query_label, train_label, n = 30, z_query = None, z_train = None, knn_graph = None):
    """\
    Description:
    ------------
        Label transfer accuracy
    """
    if knn_graph is not None:
        n_query = knn_graph.shape[0]
        n_train = knn_graph.shape[1]
        assert len(query_label) == n_query
        assert len(train_label) == n_train
        count = 0
        for i_query in range(n_query):
            neighs = np.where(knn_graph[i_query, :] != 0)[0]
            neigh_labels, neigh_votes = np.unique(train_label[neighs], return_counts = True)
            query_predict = neigh_labels[np.argmax(neigh_votes)]
            if query_predict == query_label[i_query]:
                count += 1
    else:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(z_train, train_label)
        query_predict = knn.predict(z_query)
        # np.savetxt("query_predict.txt", query_predict)
        count = 0
        for label1, label2 in zip(query_predict, query_label):
            if label1 == label2:
                count += 1
    return count / len(query_label)

########################################################################################
#
# ARI score from scIB(https://github.com/theislab/scib/tree/main/scib)
#
########################################################################################

def ari(group1, group2, implementation=None):
    """\
    Description:
    ------------
        Calculate ARI score.
    Parameters:
    ------------
        group1: labels 1
        group2: labels 2
        implementation: of set to 'sklearn', uses sklearns implementation,
        otherwise native implementation is taken
    Return:
    ------------
        ARI score
    """

    if len(group1) != len(group2):
        raise ValueError(
            f'different lengths in group1 ({len(group1)}) and group2 ({len(group2)})'
        )

    if implementation == 'sklearn':
        return adjusted_rand_score(group1, group2)

    def binom_sum(x, k=2):
        return scipy.special.binom(x, k).sum()

    n = len(group1)
    contingency = pd.crosstab(group1, group2)

    ai_sum = binom_sum(contingency.sum(axis=0))
    bi_sum = binom_sum(contingency.sum(axis=1))

    index = binom_sum(np.ravel(contingency))
    expected_index = ai_sum * bi_sum / binom_sum(n, 2)
    max_index = 0.5 * (ai_sum + bi_sum)

    return (index - expected_index) / (max_index - expected_index)



########################################################################################
#
# F1 score for rare cell type detection
#
########################################################################################
def F1_score(gt, predict):
    """\
    Description:
    ------------
        Calculation of F1 score for rare cell type detection.
    Parameters:
    ------------
        gt: ground truth labels
        predict: predicted labels
    Return:
    ------------
        F1 score
    """
    tp = np.sum((predict != 0) * (gt != 0))
    fp = np.sum((predict != 0) * (gt == 0))
    tn = np.sum((predict == 0) * (gt == 0))
    fn = np.sum((predict == 0) * (gt != 0))
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    return 2 * precision * recall / (precision + recall)


########################################################################################
#
# NMI score from scIB(https://github.com/theislab/scib/tree/main/scib)
#
########################################################################################
def nmi(group1, group2, method="arithmetic", nmi_dir=None):
    """\
    Description:
    ------------
        Calculate NMI score.
    Parameters:
    ------------
        group1: labels 1
        group2: labels 2
        method: NMI implementation
        'max': scikit method with `average_method='max'`
        'min': scikit method with `average_method='min'`
        'geometric': scikit method with `average_method='geometric'`
        'arithmetic': scikit method with `average_method='arithmetic'`
        'Lancichinetti': implementation by A. Lancichinetti 2009 et al. https://sites.google.com/site/andrealancichinetti/mutual
        'ONMI': implementation by Aaron F. McDaid et al. https://github.com/aaronmcdaid/Overlapping-NMI
        nmi_dir: directory of compiled C code if 'Lancichinetti' or 'ONMI' are specified as `method`.
        These packages need to be compiled as specified in the corresponding READMEs.
    Return:
    ------------
        NMI score
    """
    
    if len(group1) != len(group2):
        raise ValueError(
            f'different lengths in group1 ({len(group1)}) and group2 ({len(group2)})'
        )

    # choose method
    if method in ['max', 'min', 'geometric', 'arithmetic']:
        nmi_value = normalized_mutual_info_score(group1, group2, average_method=method)
    elif method == "Lancichinetti":
        nmi_value = nmi_Lanc(group1, group2, nmi_dir=nmi_dir)
    elif method == "ONMI":
        nmi_value = onmi(group1, group2, nmi_dir=nmi_dir)
    else:
        raise ValueError(f"Method {method} not valid")

    return nmi_value


def onmi(group1, group2, nmi_dir=None, verbose=True):
    """
    Based on implementation https://github.com/aaronmcdaid/Overlapping-NMI
    publication: Aaron F. McDaid, Derek Greene, Neil Hurley 2011
    params:
        nmi_dir: directory of compiled C code
    """

    if nmi_dir is None:
        raise FileNotFoundError(
            "Please provide the directory of the compiled C code from "
            "https://sites.google.com/site/andrealancichinetti/mutual3.tar.gz"
        )

    group1_file = write_tmp_labels(group1, to_int=False)
    group2_file = write_tmp_labels(group2, to_int=False)

    nmi_call = subprocess.Popen(
        [nmi_dir + "onmi", group1_file, group2_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    stdout, stderr = nmi_call.communicate()
    if stderr:
        print(stderr)

    nmi_out = stdout.decode()
    if verbose:
        print(nmi_out)

    nmi_split = [x.strip().split('\t') for x in nmi_out.split('\n')]
    nmi_max = float(nmi_split[0][1])

    # remove temporary files
    os.remove(group1_file)
    os.remove(group2_file)

    return nmi_max


def nmi_Lanc(group1, group2, nmi_dir="external/mutual3/", verbose=True):
    """
    paper by A. Lancichinetti 2009
    https://sites.google.com/site/andrealancichinetti/mutual
    recommended by Malte
    """

    if nmi_dir is None:
        raise FileNotFoundError(
            "Please provide the directory of the compiled C code from https://sites.google.com/site/andrealancichinetti/mutual3.tar.gz")

    group1_file = write_tmp_labels(group1, to_int=False)
    group2_file = write_tmp_labels(group2, to_int=False)

    nmi_call = subprocess.Popen(
        [nmi_dir + "mutual", group1_file, group2_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    stdout, stderr = nmi_call.communicate()
    if stderr:
        print(stderr)
    nmi_out = stdout.decode().strip()

    return float(nmi_out.split('\t')[1])


def write_tmp_labels(group_assignments, to_int=False, delim='\n'):
    """
    write the values of a specific obs column into a temporary file in text format
    needed for external C NMI implementations (onmi and nmi_Lanc functions), because they require files as input
    params:
        to_int: rename the unique column entries by integers in range(1,len(group_assignments)+1)
    """
    import tempfile

    if to_int:
        label_map = {}
        i = 1
        for label in set(group_assignments):
            label_map[label] = i
            i += 1
        labels = delim.join([str(label_map[name]) for name in group_assignments])
    else:
        labels = delim.join([str(name) for name in group_assignments])

    clusters = {label: [] for label in set(group_assignments)}
    for i, label in enumerate(group_assignments):
        clusters[label].append(str(i))

    output = '\n'.join([' '.join(c) for c in clusters.values()])
    output = str.encode(output)

    # write to file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(output)
        filename = f.name

    return filename




########################################################################################
#
# silhouette score from scIB(https://github.com/theislab/scib/tree/main/scib), silhouette
# score measures the differences between within cluster distances and between cluster distances
# for cell types, larger silhouette score means better result, and for batches smaller 
# silhouette score means better result. 
# 
# FUNCTIONS:
# -------------------------------------------------
# silhouette: calculate the cell type separation
# silhouette_batch: calculate the batch mixing
# -------------------------------------------------
#
########################################################################################

def silhouette(
        X,
        group_gt,
        metric='euclidean',
        scale=True
):
    """\
    Description:
    ------------
        Wrapper for sklearn silhouette function values range from [-1, 1] with
            1 being an ideal fit
            0 indicating overlapping clusters and
            -1 indicating misclassified cells
        By default, the score is scaled between 0 and 1. This is controlled `scale=True`

    Parameters:
    ------------
        group_gt: cell labels
        X: embedding 
        scale: default True, scale between 0 (worst) and 1 (best)
        metric: distance metric
    """
    asw = silhouette_score(
        X=X,
        labels=group_gt,
        metric=metric
    )
    if scale:
        asw = (asw + 1) / 2
    return asw


def silhouette_batch(
        X,
        batch_gt,
        group_gt,
        metric='euclidean',
        return_all=False,
        scale=True,
        verbose=True
):
    """\
    Description:
    ------------
        Absolute silhouette score of batch labels subsetted for each group.

    Parameters:
    -----------
        batch_gt: ground truth batch label
        group_gt: ground truth group label
        X: embedding
        metric: see sklearn silhouette score
        scale: if True, scale between 0 and 1
        return_all: if True, return all silhouette scores and label means
        default False: return average width silhouette (ASW)
        verbose:
    
    Returns:
    -----------
        average width silhouette ASW
        mean silhouette per group in pd.DataFrame
        Absolute silhouette scores per group label
    """

    sil_all = pd.DataFrame(columns=['group', 'silhouette_score'])

    for group in np.sort(np.unique(group_gt)):
        X_group = X[group_gt == group, :]
        batch_group = batch_gt[group_gt == group]
        n_batches = np.unique(batch_group).shape[0]

        if (n_batches == 1) or (n_batches == X_group.shape[0]):
            continue

        sil_per_group = silhouette_samples(
            X_group,
            batch_group,
            metric=metric
        )

        # take only absolute value
        sil_per_group = [abs(i) for i in sil_per_group]

        if scale:
            # scale s.t. highest number is optimal
            sil_per_group = [1 - i for i in sil_per_group]

        sil_all = sil_all.append(
            pd.DataFrame({
                'group': [group] * len(sil_per_group),
                'silhouette_score': sil_per_group
            })
        )

    sil_all = sil_all.reset_index(drop=True)
    sil_means = sil_all.groupby('group').mean()
    asw = sil_means['silhouette_score'].mean()

    if verbose:
        print(f'mean silhouette per cell: {sil_means}')

    if return_all:
        return asw, sil_means, sil_all

    return asw




#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#
# Batch effect removal per cell identity label (consider the batch mixing for each cell type)
#
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

########################################################################################
#
# graph connectivity score from scIB (https://github.com/theislab/scib/tree/main/scib), 
# It measures the mixing of the batches, and assesses whether the kNN graph 
# representation, ​G,​ of the integrated data directly connects all cells with 
# the same cell identity label. It first construct a KNN graph on the integrated latent 
# embedding, then it calculate that for cells of each cell type, how well the subgraph
# is connected. 
# 
# FUNCTIONS:
# -------------------------------------------------
# def graph_connectivity(adata, label_key):
#     """"
#     Quantify how connected the subgraph corresponding to each batch cluster is.
#     Calculate per label: #cells_in_largest_connected_component/#all_cells
#     Final score: Average over labels

#     :param adata: adata with computed neighborhood graph
#     :param label_key: name in adata.obs containing the cell identity labels
#     """
#     if 'neighbors' not in adata.uns:
#         raise KeyError(
#             'Please compute the neighborhood graph before running this function!'
#         )

#     clust_res = []

#     for label in adata.obs[label_key].cat.categories:
#         adata_sub = adata[adata.obs[label_key].isin([label])]
#         _, labels = connected_components(
#             adata_sub.obsp['connectivities'],
#             connection='strong'
#         )
#         tab = pd.value_counts(labels)
#         clust_res.append(tab.max() / sum(tab))

#     return np.mean(clust_res)
# -------------------------------------------------
########################################################################################

def graph_connectivity(X = None, G = None, groups = None, k = 10):
    """\
    Description:
    ------------
        Quantify how connected the subgraph corresponding to each batch cluster is.
        Calculate per label: #cells_in_largest_connected_component/#all_cells
        Final score: Average over labels

    Parameters:
    ------------
        X: embedding
        G: embedding graph
        groups: group labels
        k: number of neighbors
    """
    clust_res = []
    if X is not None: 
        # calculate the adjacency matrix of the neighborhood graph
        G = kneighbors_graph(X, n_neighbors = k, mode='connectivity', include_self = False)
    elif G is None:
        raise ValueError("Either X or G should be provided")
    
    # make sure all cells have labels
    assert groups.shape[0] == G.shape[0]
    
    for group in np.sort(np.unique(groups)):
        G_sub = G[groups == group,:][:, groups == group]
        # print(G_sub.shape)
        _, labels = connected_components(csr_matrix(G_sub), connection='strong')
        tab = pd.value_counts(labels)
        clust_res.append(tab.max() / sum(tab))

    return np.mean(clust_res)


#-------------------------------------------------------------------------------------------
#
# statsmodel
#
#-------------------------------------------------------------------------------------------
def _ecdf(x):
    '''no frills empirical cdf used in fdrcorrection
    '''
    nobs = len(x)
    return np.arange(1,nobs+1)/float(nobs)

def fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False):
    '''
    pvalue correction for false discovery rate.

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.

    Parameters
    ----------
    pvals : array_like, 1d
        Set of p-values of the individual tests.
    alpha : float, optional
        Family-wise error rate. Defaults to ``0.05``.
    method : {'i', 'indep', 'p', 'poscorr', 'n', 'negcorr'}, optional
        Which method to use for FDR correction.
        ``{'i', 'indep', 'p', 'poscorr'}`` all refer to ``fdr_bh``
        (Benjamini/Hochberg for independent or positively
        correlated tests). ``{'n', 'negcorr'}`` both refer to ``fdr_by``
        (Benjamini/Yekutieli for general or negatively correlated tests).
        Defaults to ``'indep'``.
    is_sorted : bool, optional
        If False (default), the p_values will be sorted, but the corrected
        pvalues are in the original order. If True, then it assumed that the
        pvalues are already sorted in ascending order.

    Returns
    -------
    rejected : ndarray, bool
        True if a hypothesis is rejected, False if not
    pvalue-corrected : ndarray
        pvalues adjusted for multiple hypothesis testing to limit FDR

    Notes
    -----
    If there is prior information on the fraction of true hypothesis, then alpha
    should be set to ``alpha * m/m_0`` where m is the number of tests,
    given by the p-values, and m_0 is an estimate of the true hypothesis.
    (see Benjamini, Krieger and Yekuteli)

    The two-step method of Benjamini, Krieger and Yekutiel that estimates the number
    of false hypotheses will be available (soon).

    Both methods exposed via this function (Benjamini/Hochberg, Benjamini/Yekutieli)
    are also available in the function ``multipletests``, as ``method="fdr_bh"`` and
    ``method="fdr_by"``, respectively.

    See also
    --------
    multipletests

    '''
    pvals = np.asarray(pvals)
    assert pvals.ndim == 1, "pvals must be 1-dimensional, that is of shape (n,)"

    if not is_sorted:
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
    else:
        pvals_sorted = pvals  # alias

    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1./np.arange(1, len(pvals_sorted)+1))   #corrected this
        ecdffactor = _ecdf(pvals_sorted) / cm
##    elif method in ['n', 'negcorr']:
##        cm = np.sum(np.arange(len(pvals)))
##        ecdffactor = ecdf(pvals_sorted)/cm
    else:
        raise ValueError('only indep and negcorr implemented')
    reject = pvals_sorted <= ecdffactor*alpha
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
        reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected>1] = 1
    if not is_sorted:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        del pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_
    else:
        return reject, pvals_corrected

def wilcoxon_rank_sum(counts_x, counts_y, fdr = False):
    # wilcoxon rank sum test
    ngenes = counts_x.shape[1]
    assert ngenes == counts_y.shape[1]
    pvals = []
    for gene_i in range(ngenes):
        _, pval = stats.ranksums(x = counts_x[:, gene_i].squeeze(), y = counts_y[:, gene_i].squeeze())
        pvals.append(pval)
    pvals = np.array(pvals)

    # fdr correction
    if fdr:
        _, pvals = fdrcorrection(pvals)
    return pvals