import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from itertools import product, permutations, combinations, combinations_with_replacement

from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay,roc_curve,auc,RocCurveDisplay, average_precision_score, roc_auc_score

def kendalltau(G_inf, G_true):
    """\
    Description:
    ------------
        Pearson Correlation
    """
    import scipy.stats as stats
    return stats.kendalltau(G_inf.reshape(-1), G_true.reshape(-1))
    
def NMSE(G_inf, G_true):
    """\
    Description:
    ------------
        Normalized Mean Square Error, in GLAD paper, used for two real value matrices
    """
    return ((G_inf - G_true) ** 2).sum()/(G_true ** 2).sum()

def PS(G_inf, G_true):
    """\
    Description:
    ------------
        Probability of success, sign match, from GLAD paper
    """
    return int(np.all(np.sign(G_inf)==np.sign(G_true)))

def pearson(G_inf, G_true):
    """\
    Description:
    ------------
        Pearson Correlation
    """
    import scipy.stats as stats
    return stats.pearsonr(G_inf.reshape(-1), G_true.reshape(-1))

def spearman(G_inf, G_true):
    """\
    Description:
    ------------
        Pearson Correlation
    """
    import scipy.stats as stats
    return stats.spearmanr(G_inf.reshape(-1), G_true.reshape(-1))

def cossim(G_inf, G_true):
    """\
    Description:
    ------------
        Cosine similarity
    """
    G_inf_norm = np.sqrt((G_inf.reshape(-1) ** 2).sum())
    G_true_norm = np.sqrt((G_true.reshape(-1) ** 2).sum())
    return np.sum(G_inf.reshape(-1) * G_true.reshape(-1))/G_inf_norm/G_true_norm

def compute_auc_signed(G_inf, G_true):
    # assert np.allclose(G_inf, G_inf.T, atol = 1e-7)
    # assert np.allclose(G_true, G_true.T, atol = 1e-7)
    gt_pos = G_true.copy()
    estm_pos = G_inf.copy()
    gt_pos[np.where(G_true < 0)] = 0
    estm_pos[np.where(estm_pos < 0)] = 0
    # binarize
    gt_pos = (gt_pos > 1e-6).astype(int)
    estm_pos = (estm_pos - np.min(estm_pos))/(np.max(estm_pos) - np.min(estm_pos) + 1e-12)
    _, _, _, _, AUPRC_pos, AUROC_pos, _ = _compute_auc(estm_pos, gt_pos)

    gt_neg = G_true.copy()
    estm_neg = G_inf.copy()
    gt_neg[np.where(G_true > 0)] = 0
    estm_neg[np.where(estm_neg > 0)] = 0
    # binarize
    gt_neg = (gt_neg < -1e-6).astype(int)
    estm_neg = - estm_neg
    estm_neg = (estm_neg - np.min(estm_neg))/(np.max(estm_neg) - np.min(estm_neg) + 1e-12)
    _, _, _, _, AUPRC_neg, AUROC_neg, _ = _compute_auc(estm_neg, gt_neg)

    return AUPRC_pos, AUPRC_neg

def compute_auc_abs(G_inf, G_true):
    # assert np.allclose(G_inf, G_inf.T, atol = 1e-7)
    # assert np.allclose(G_true, G_true.T, atol = 1e-7)
    G_inf_abs = np.abs(G_inf)
    G_true_abs = np.abs(G_true)
    G_true_abs = (G_true_abs > 1e-6).astype(int)
    G_inf_abs = (G_inf_abs - np.min(G_inf_abs))/(np.max(G_inf_abs) - np.min(G_inf_abs) + 1e-12)
    _, _, _, _, AUPRC, AUROC, _ = _compute_auc(G_inf_abs, G_true_abs)
    return AUPRC

def _compute_auc(estm_adj, gt_adj):
    """\
    Description:
    ------------
        calculate AUPRC and AUROC
    Parameters:
    ------------
        estm_adj: predict graph adjacency matrix
        gt_adj: ground truth graph adjacency matrix
        directed: the directed estimation or not
    Return:
    ------------
        prec: precision
        recall: recall
        fpr: false positive rate
        tpr: true positive rate
        AUPRC, AUROC
    """
    
    if np.max(estm_adj) == 0:
        return 0, 0, 0, 0, 0, 0, 0
    else:
        fpr, tpr, thresholds = roc_curve(y_true=gt_adj.reshape(-1,), y_score=estm_adj.reshape(-1,), pos_label=1)
        
        if len(set(gt_adj.reshape(-1,))) == 1:
            prec, recall = np.array([0., 1.]), np.array([1., 0.])
        else:
            prec, recall, thresholds = precision_recall_curve(y_true=gt_adj.reshape(-1,), probas_pred=estm_adj.reshape(-1,), pos_label=1)

        # the same
        # AUPRC = average_precision_score(gt_adj.reshape(-1,), estm_adj.reshape(-1,)) 
        return prec, recall, fpr, tpr, auc(recall, prec), auc(fpr, tpr), thresholds    

def compute_auc(estm_adj, gt_adj, directed = False):
    """\
    Description:
    ------------
        calculate AUPRC and AUROC
    Parameters:
    ------------
        estm_adj: predict graph adjacency matrix
        gt_adj: ground truth graph adjacency matrix
        directed: the directed estimation or not
    Return:
    ------------
        prec: precision
        recall: recall
        fpr: false positive rate
        tpr: true positive rate
        AUPRC, AUROC
    """
    # make absolute and within (0,1)
    estm_norm_adj = np.abs(estm_adj)/np.max(np.abs(estm_adj) + 1e-12)
    
    if np.max(estm_norm_adj) == 0:
        return 0, 0, 0, 0, 0, 0, 0
    else:
        # assert np.abs(np.max(estm_norm_adj) - 1) < 1e-4
        if directed == False:
            gt_adj = ((gt_adj + gt_adj.T) > 0).astype(int)
        np.fill_diagonal(gt_adj, 0)
        np.fill_diagonal(estm_norm_adj, 0)
        rows, cols = np.where(gt_adj != 0)

        fpr, tpr, thresholds = roc_curve(y_true=gt_adj.reshape(-1,), y_score=estm_norm_adj.reshape(-1,), pos_label=1)
        
        ngenes, _ = gt_adj.shape
        
        if len(set(gt_adj.reshape(-1,))) == 1:
            prec, recall = np.array([0., 1.]), np.array([1., 0.])
        else:
            prec, recall, thresholds = precision_recall_curve(y_true=gt_adj.reshape(-1,), probas_pred=estm_norm_adj.reshape(-1,), pos_label=1)

        # the same
        # AUPRC = average_precision_score(gt_adj.reshape(-1,), estm_norm_adj.reshape(-1,)) 
        return prec, recall, fpr, tpr, auc(recall, prec), auc(fpr, tpr), thresholds    

def compute_auc_ori(estm_adj, gt_adj, directed = False):
    """\
    Description:
    ------------
        calculate AUPRC and AUROC
    Parameters:
    ------------
        estm_adj: predict graph adjacency matrix
        gt_adj: ground truth graph adjacency matrix
        directed: the directed estimation or not
    Return:
    ------------
        prec: precision
        recall: recall
        fpr: false positive rate
        tpr: true positive rate
        AUPRC, AUROC
    """
    # make symmetric
    estm_norm_adj = np.abs(estm_adj)/np.max(np.abs(estm_adj) + 1e-12)
    
    # estm_adj = (estm_adj - np.min(estm_adj))/(np.max(estm_adj) - np.min(estm_adj))
    if np.max(estm_norm_adj) == 0:
        return 0, 0, 0, 0, 0, 0
    else:
        # assert np.abs(np.max(estm_norm_adj) - 1) < 1e-4
        if directed == False:
            gt_adj = ((gt_adj + gt_adj.T) > 0).astype(np.int)
        np.fill_diagonal(gt_adj, 0)
        np.fill_diagonal(estm_norm_adj, 0)
        rows, cols = np.where(gt_adj != 0)

        trueEdgesDF = pd.DataFrame(columns = ["Gene1", "Gene2", "EdgeWeight"])
        trueEdgesDF.Gene1 = [str(x) for x in rows]
        trueEdgesDF.Gene2 = [str(y) for y in cols]
        trueEdgesDF.EdgeWeight = 1

        rows, cols = np.where(estm_norm_adj != 0)
        predEdgeDF = pd.DataFrame(columns = ["Gene1", "Gene2", "EdgeWeight"])
        predEdgeDF.Gene1 = [str(x) for x in rows]
        predEdgeDF.Gene2 = [str(y) for y in cols]
        predEdgeDF.EdgeWeight = np.array([estm_norm_adj[i,j] for i,j in zip(rows,cols)])

        # order according to ranks
        order = np.argsort(predEdgeDF.EdgeWeight.values.squeeze())[::-1]
        predEdgeDF = predEdgeDF.iloc[order,:]

        prec, recall, fpr, tpr, AUPRC, AUROC = _computeScores(trueEdgesDF, predEdgeDF, directed = directed, selfEdges = False)

        return prec, recall, fpr, tpr, AUPRC, AUROC



def _computeScores(trueEdgesDF, predEdgeDF, 
directed = True, selfEdges = True):
    '''        
    Computes precision-recall and ROC curves
    using scikit-learn for a given set of predictions in the 
    form of a DataFrame.
    
    :param trueEdgesDF:   A pandas dataframe containing the true classes.The indices of this dataframe are all possible edgesin a graph formed using the genes in the given dataset. This dataframe only has one column to indicate the classlabel of an edge. If an edge is present in the reference network, it gets a class label of 1, else 0.
    :type trueEdgesDF: DataFrame
        
    :param predEdgeDF:   A pandas dataframe containing the edge ranks from the prediced network. The indices of this dataframe are all possible edges.This dataframe only has one column to indicate the edge weightsin the predicted network. Higher the weight, higher the edge confidence.
    :type predEdgeDF: DataFrame
    
    :param directed:   A flag to indicate whether to treat predictionsas directed edges (directed = True) or undirected edges (directed = False).
    :type directed: bool
    :param selfEdges:   A flag to indicate whether to includeself-edges (selfEdges = True) or exclude self-edges (selfEdges = False) from evaluation.
    :type selfEdges: bool
        
    :returns:
            - prec: A list of precision values (for PR plot)
            - recall: A list of precision values (for PR plot)
            - fpr: A list of false positive rates (for ROC plot)
            - tpr: A list of true positive rates (for ROC plot)
            - AUPRC: Area under the precision-recall curve
            - AUROC: Area under the ROC curve
    '''

    if directed:        
        # Initialize dictionaries with all 
        # possible edges
        if selfEdges:
            possibleEdges = list(product(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                         repeat = 2))
        else:
            possibleEdges = list(permutations(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                         r = 2))
        
        TrueEdgeDict = {'|'.join(p):0 for p in possibleEdges}
        PredEdgeDict = {'|'.join(p):0 for p in possibleEdges}
        
        # Compute TrueEdgeDict Dictionary
        # 1 if edge is present in the ground-truth
        # 0 if edge is not present in the ground-truth
        for key in TrueEdgeDict.keys():
            if len(trueEdgesDF.loc[(trueEdgesDF['Gene1'] == key.split('|')[0]) &
                   (trueEdgesDF['Gene2'] == key.split('|')[1])])>0:
                    TrueEdgeDict[key] = 1
                
        for key in PredEdgeDict.keys():
            subDF = predEdgeDF.loc[(predEdgeDF['Gene1'] == key.split('|')[0]) &
                               (predEdgeDF['Gene2'] == key.split('|')[1])]
            if len(subDF)>0:
                PredEdgeDict[key] = np.abs(subDF.EdgeWeight.values[0])

    # if undirected
    else:
        # Initialize dictionaries with all 
        # possible edges
        if selfEdges:
            possibleEdges = list(combinations_with_replacement(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                                               r = 2))
        else:
            possibleEdges = list(combinations(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                                               r = 2))
        TrueEdgeDict = {'|'.join(p):0 for p in possibleEdges}
        PredEdgeDict = {'|'.join(p):0 for p in possibleEdges}

        # Compute TrueEdgeDict Dictionary
        # 1 if edge is present in the ground-truth
        # 0 if edge is not present in the ground-truth

        for key in TrueEdgeDict.keys():
            if len(trueEdgesDF.loc[((trueEdgesDF['Gene1'] == key.split('|')[0]) &
                           (trueEdgesDF['Gene2'] == key.split('|')[1])) |
                              ((trueEdgesDF['Gene2'] == key.split('|')[0]) &
                           (trueEdgesDF['Gene1'] == key.split('|')[1]))]) > 0:
                TrueEdgeDict[key] = 1  

        # Compute PredEdgeDict Dictionary
        # from predEdgeDF

        for key in PredEdgeDict.keys():
            subDF = predEdgeDF.loc[((predEdgeDF['Gene1'] == key.split('|')[0]) &
                               (predEdgeDF['Gene2'] == key.split('|')[1])) |
                              ((predEdgeDF['Gene2'] == key.split('|')[0]) &
                               (predEdgeDF['Gene1'] == key.split('|')[1]))]
            if len(subDF)>0:
                PredEdgeDict[key] = max(np.abs(subDF.EdgeWeight.values))

                
                
    # Combine into one dataframe
    # to pass it to sklearn
    outDF = pd.DataFrame([TrueEdgeDict,PredEdgeDict]).T
    outDF.columns = ['TrueEdges','PredEdges']
    
    fpr, tpr, thresholds = roc_curve(y_true=outDF['TrueEdges'],
                                     y_score=outDF['PredEdges'], pos_label=1)

    prec, recall, thresholds = precision_recall_curve(y_true=outDF['TrueEdges'],
                                                      probas_pred=outDF['PredEdges'], pos_label=1)
    
    return prec, recall, fpr, tpr, auc(recall, prec), auc(fpr, tpr)



def compute_earlyprec(estm_adj, gt_adj, directed = False, TFEdges = False):
    """\
    Description:
    ------------
        Calculate the early precision ratio. 
        Early precision: the fraction of true positives in the top-k edges. 
        Early precision ratio: the ratio of the early precision between estim and random estim.
        directed: the directed estimation or not
    Parameters:
    ------------
        estm_adj: estimated adjacency matrix
        gt_adj: ground truth adjacency matrix
        TFEdges: use transcription factor
    
    """
    estm_norm_adj = np.abs(estm_adj)/np.max(np.abs(estm_adj) + 1e-12)
    if np.max(estm_norm_adj) == 0:
        return 0, 0
    else:
        # estm_adj = (estm_adj - np.min(estm_adj))/(np.max(estm_adj) - np.min(estm_adj))
        
        # assert np.abs(np.max(estm_norm_adj) - 1) < 1e-4
        # if directed == False:
        #     gt_adj = ((gt_adj + gt_adj.T) > 0).astype(np.int)
        # np.fill_diagonal(gt_adj, 0)
        # np.fill_diagonal(estm_norm_adj, 0)
        rows, cols = np.where(gt_adj != 0)

        trueEdgesDF = pd.DataFrame(columns = ["Gene1", "Gene2", "EdgeWeight"])
        trueEdgesDF.Gene1 = np.array([str(x) for x in rows], dtype = np.object)
        trueEdgesDF.Gene2 = np.array([str(y) for y in cols], dtype = np.object)
        trueEdgesDF.EdgeWeight = 1

        rows, cols = np.where(estm_norm_adj != 0)
        predEdgeDF = pd.DataFrame(columns = ["Gene1", "Gene2", "EdgeWeight"])
        predEdgeDF.Gene1 = np.array([str(x) for x in rows], dtype = np.object)
        predEdgeDF.Gene2 = np.array([str(y) for y in cols], dtype = np.object)
        predEdgeDF.EdgeWeight = np.array([estm_norm_adj[i,j] for i,j in zip(rows,cols)])

        # order according to ranks
        order = np.argsort(predEdgeDF.EdgeWeight.values.squeeze())[::-1]
        predEdgeDF = predEdgeDF.iloc[order,:]


        trueEdgesDF = trueEdgesDF.loc[(trueEdgesDF['Gene1'] != trueEdgesDF['Gene2'])]
        trueEdgesDF.drop_duplicates(keep = 'first', inplace=True)
        trueEdgesDF.reset_index(drop=True, inplace=True)


        predEdgeDF = predEdgeDF.loc[(predEdgeDF['Gene1'] != predEdgeDF['Gene2'])]
        predEdgeDF.drop_duplicates(keep = 'first', inplace=True)
        predEdgeDF.reset_index(drop=True, inplace=True)

        if TFEdges:
            # Consider only edges going out of TFs

            # Get a list of all possible TF to gene interactions 
            uniqueNodes = np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']])
            possibleEdges_TF = set(product(set(trueEdgesDF.Gene1),set(uniqueNodes)))

            # Get a list of all possible interactions 
            possibleEdges_noSelf = set(permutations(uniqueNodes, r = 2))

            # Find intersection of above lists to ignore self edges
            # TODO: is there a better way of doing this?
            possibleEdges = possibleEdges_TF.intersection(possibleEdges_noSelf)

            TrueEdgeDict = {'|'.join(p):0 for p in possibleEdges}

            trueEdges = trueEdgesDF['Gene1'] + "|" + trueEdgesDF['Gene2']
            trueEdges = trueEdges[trueEdges.isin(TrueEdgeDict)]
            print("\nEdges considered ", len(trueEdges))
            numEdges = len(trueEdges)

            predEdgeDF['Edges'] = predEdgeDF['Gene1'] + "|" + predEdgeDF['Gene2']
            # limit the predicted edges to the genes that are in the ground truth
            predEdgeDF = predEdgeDF[predEdgeDF['Edges'].isin(TrueEdgeDict)]

        else:
            trueEdges = trueEdgesDF['Gene1'] + "|" + trueEdgesDF['Gene2']
            trueEdges = set(trueEdges.values)
            numEdges = len(trueEdges)

        # check if ranked edges list is empty
        # if so, it is just set to an empty set
        if not predEdgeDF.shape[0] == 0:

            # we want to ensure that we do not include
            # edges without any edge weight
            # so check if the non-zero minimum is
            # greater than the edge weight of the top-kth
            # node, else use the non-zero minimum value.
            predEdgeDF.EdgeWeight = predEdgeDF.EdgeWeight.round(6)
            predEdgeDF.EdgeWeight = predEdgeDF.EdgeWeight.abs()

            # Use num True edges or the number of
            # edges in the dataframe, which ever is lower
            maxk = min(predEdgeDF.shape[0], numEdges)
            # find the maxkth edge weight
            edgeWeightTopk = predEdgeDF.iloc[maxk-1].EdgeWeight

            # find the smallest non-zero edge weight
            nonZeroMin = np.nanmin(predEdgeDF.EdgeWeight.replace(0, np.nan).values)

            # choose the largest one from nonZeroMin and edgeWeightTopk
            bestVal = max(nonZeroMin, edgeWeightTopk)

            # find all the edges with edge weight larger than the bestVal
            newDF = predEdgeDF.loc[(predEdgeDF['EdgeWeight'] >= bestVal)]
            # rankDict is a set that stores all significant edges
            rankDict = set(newDF['Gene1'] + "|" + newDF['Gene2'])
        else:
            # raise ValueError("No prediction")
            rankDict = []

        if len(rankDict) != 0:
            intersectionSet = rankDict.intersection(trueEdges)
            Eprec = len(intersectionSet)/(len(rankDict)+1e-12)
            Erec = len(intersectionSet)/(len(trueEdges)+1e-12)
        else:
            Eprec = 0
            Erec = 0

    return Eprec, Erec


def compute_eprec_signed(G_inf, G_true):
    # assert np.allclose(G_inf, G_inf.T, atol = 1e-7)
    # assert np.allclose(G_true, G_true.T, atol = 1e-7)
    gt_pos = G_true.copy()
    estm_pos = G_inf.copy()
    gt_pos[np.where(G_true < 0)] = 0
    estm_pos[np.where(estm_pos < 0)] = 0
    # binarize
    gt_pos = (gt_pos > 1e-6).astype(int)
    estm_pos = (estm_pos - np.min(estm_pos))/(np.max(estm_pos) - np.min(estm_pos) + 1e-12)
    Eprec_pos, Erec_pos = compute_earlyprec(estm_pos, gt_pos)

    gt_neg = G_true.copy()
    estm_neg = G_inf.copy()
    gt_neg[np.where(G_true > 0)] = 0
    estm_neg[np.where(estm_neg > 0)] = 0
    # binarize
    gt_neg = (gt_neg < -1e-6).astype(int)
    estm_neg = - estm_neg
    estm_neg = (estm_neg - np.min(estm_neg))/(np.max(estm_neg) - np.min(estm_neg) + 1e-12)
    Eprec_neg, Erec_neg = compute_earlyprec(estm_neg, gt_neg)

    return Eprec_pos, Eprec_neg

def compute_eprec_abs(G_inf, G_true):
    # assert np.allclose(G_inf, G_inf.T, atol = 1e-7)
    # assert np.allclose(G_true, G_true.T, atol = 1e-7)
    G_inf_abs = np.abs(G_inf)
    G_true_abs = np.abs(G_true)
    G_true_abs = (G_true_abs > 1e-6).astype(int)
    G_inf_abs = (G_inf_abs - np.min(G_inf_abs))/(np.max(G_inf_abs) - np.min(G_inf_abs) + 1e-12)
    Eprec, Erec = compute_earlyprec(G_inf_abs, G_true_abs)
    return Eprec