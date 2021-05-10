import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_selection import  SelectFdr,SelectPercentile,f_classif
from numpy import linalg as LA
import math
import argparse
import itertools
import scipy.io as scio
import pandas as pd
import time
import scipy.stats as stats
from statsmodels.stats.weightstats import ttest_ind
from scipy  import sparse
import utils

def quantileNormalize(df_input):
    df = df_input.copy()
    #compute rank
    dic = {}
    for col in df:
        dic.update({col : sorted(df[col])})
    sorted_df = pd.DataFrame(dic)
    rank = sorted_df.mean(axis = 1).tolist()
    #sort
    for col in df:
        t = np.searchsorted(np.sort(df[col]), df[col])
        df[col] = [rank[i] for i in t]
    return df

def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j


class coupleNMF():
    def __init__(self, counts, N=3, lambda1 = None, lambda2 = None):
        super().__init__()
        # runs
        self.rep = 50
        self.N = N

        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # gene by cell
        counts_rna = counts["rna"][0]
        # region by cell
        counts_atac = counts["atac"][0]
        # gene by region
        gact = counts["gact"][0]
        
        self.G = quantileNormalize(pd.DataFrame(counts_rna))
        self.G[self.G>10000] = 10000
        self.G = np.log(1+self.G)
        self.R = quantileNormalize(pd.DataFrame(counts_atac))
        self.A = utils.preprocess(gact, mode = "gact")

    def train_func(self, E_symbol, P_symbol):
        print("Initializing non-negative matrix factorization for gene expression data...")

        # start the calculation
        err1=np.zeros(self.rep)
        for i in range(0,self.rep):
            # call sklearn nmf function as initialization, repeat for rep times with different random states
            model = NMF(n_components=self.N, init='random', random_state=i,solver='cd',max_iter=50)
            # only for gene expression data
            W20 = model.fit_transform(self.G)
            H20 = model.components_
            # error l2 norm
            err1[i]=LA.norm(self.G-np.dot(W20,H20),ord = 'fro')

        # select the best one as initialization
        model = NMF(n_components=self.N, init='random', random_state=np.argmin(err1),solver='cd',max_iter=1000)
        W20 = model.fit_transform(self.G)
        H20 = model.components_
        S20=np.argmax(H20,0)

        print("Initializing non-negative matrix factorization for chromatin accessibility data...")
        # do the same process for R as for G
        self.R = np.log(self.R+1)
        err=np.zeros(self.rep)
        for i in range(0,self.rep):
            # ||X - W@H||
            model = NMF(n_components=self.N, init='random', random_state=i,solver='cd',max_iter=50)
            W10 = model.fit_transform(self.R)
            H10 = model.components_
            err[i]=LA.norm(self.R-np.dot(W10,H10),ord = 'fro')

        model = NMF(n_components=self.N, init='random', random_state=np.argmin(err),solver='cd',max_iter=1000)
        W10 = model.fit_transform(self.R)
        H10 = model.components_
        S10=np.argmax(H10,0)

        print("Selecting differentially expressed genes...")
        p2 = np.zeros((self.G.shape[0],self.N))
        for i in range(self.N):
            for j in range(self.G.shape[0]):
                statistic, p2[j,i],df  = ttest_ind(self.G.iloc[j,S20==i], self.G.iloc[j,S20!=i] ,alternative='smaller')

        # binary matrix, denote the differntially expressed gene in each cluster (with 1)
        WP2 = np.zeros((W20.shape))
        p2[np.isnan(p2) ] = 1
        scores = -np.log10(p2)
        
        # 0.05 of the total number of genes
        temp = int(len(E_symbol)/20)
        for i in range(self.N):
            indexs = scores[:,i].argsort()[-temp:][::-1]
            WP2[indexs,i] = 1

        print("Selecting differentially open peaks...")
        p1 = np.zeros((self.R.shape[0],self.N))
        for i in range(self.N):
            for j in range(self.R.shape[0]):
                statistic, p1[j,i],df  = ttest_ind(self.R.iloc[j,S10==i], self.R.iloc[j,S10!=i] ,alternative='smaller')

        # binary matrix, denote the differntially expressed gene in each cluster (with 1)
        # WP1 and WP2 calculated at the beginning is used for matching the latent dimensions for init
        WP1 = np.zeros((W10.shape))
        p1[np.isnan(p1) ] = 1
        scores = -np.log10(p1)
        
        # 0.05 of the total number of regions 
        temp = int(len(P_symbol)/20)
        for i in range(self.N):
            indexs = scores[:,i].argsort()[-temp:][::-1]
            WP1[indexs,i] = 1

        # permute all possible cluster matching (gene and region)
        perm = list(itertools.permutations(range(self.N)))
        score = np.zeros(len(perm))
        for i in range(len(perm)):
            score[i] = np.trace(np.dot(np.dot(np.transpose(WP2),self.A),WP1))
        
        # find the cluster match that have the largest score
        match = np.argmax(score)
        # use that as the initialization for coupleNMF, such that the latent dimensions are matched
        W20 = W20[:,perm[match]]
        H20 = H20[perm[match],:]
        # make the loading non-negative
        S20=np.argmax(H20,0)

        print("Initializing hyperparameters lambda1, lambda2 and mu...")
        lambda10 = pow(LA.norm(self.G-np.dot(W20,H20),ord = 'fro'),2)/pow(LA.norm(self.R-np.dot(W10,H10),ord = 'fro'),2)
        lambda20 = pow(np.trace(np.dot(np.dot(np.transpose(W20),self.A),W10)),2)/pow(LA.norm(self.R-np.dot(W10,H10),ord = 'fro'),2)
        if type(self.lambda1) == type(None) and type(self.lambda2) == type(None):
            set1=[lambda10*pow(5,0),lambda10*pow(5,1),lambda10*pow(5,2),lambda10*pow(5,3),lambda10*pow(5,4)]
            set2=[lambda20*pow(5,-4),lambda20*pow(5,-3),lambda20*pow(5,-2),lambda20*pow(5,-1),lambda20*pow(5,0)]
        elif type(self.lambda1) == type(None):
            set1=[lambda10*pow(5,0),lambda10*pow(5,1),lambda10*pow(5,2),lambda10*pow(5,3),lambda10*pow(5,4)]
            set2=[self.lambda2]
        elif type(self.lambda2) == type(None):
            set1=[self.lambda1]
            set2=[lambda20*pow(5,-4),lambda20*pow(5,-3),lambda20*pow(5,-2),lambda20*pow(5,-1),lambda20*pow(5,0)]

        else:
            set1=[self.lambda1*lambda10]
            set2=[self.lambda2*lambda20]


        mu = 1
        eps = 0.001
        detr = np.zeros((len(set1),len(set2)))
        detr1 = np.zeros((len(set1),len(set2)))
        S1_all = np.zeros((len(set1)*len(set2),self.R.shape[1]))
        S2_all = np.zeros((len(set1)*len(set2),self.G.shape[1]))
        P_all = np.zeros((len(set1)*len(set2),self.N,self.R.shape[0]))
        E_all = np.zeros((len(set1)*len(set2),self.N,self.G.shape[0]))
        P_p_all = np.zeros((len(set1)*len(set2),self.N,self.R.shape[0]))
        E_p_all = np.zeros((len(set1)*len(set2),self.N,self.G.shape[0]))
        print("Starting coupleNMF...")
        count = 0

        # scan through all hyper-parameters if not given
        for x in range(len(set1)):
            for y in range(len(set2)):
                lambda1 = set1[x]
                lambda2 = set2[y]
                # initialize
                W1 = W10
                W2 = W20
                H1 = H10
                H2 = H20
                print(lambda1,lambda2)
            
                
                print("Iterating coupleNMF...")
                maxiter   = 500
                err       = 1
                terms     = np.zeros(maxiter)
                it        = 0
                terms[it] = lambda1*pow(LA.norm(self.G-np.dot(W2,H2),ord = 'fro'),2)+pow(LA.norm(self.R-np.dot(W1,H1),ord = 'fro'),2)+lambda2*pow(np.trace(np.dot(np.dot(np.transpose(W2),self.A),W1)),2)+mu*(pow(LA.norm(W1,ord = 'fro'),2)+pow(LA.norm(W2,ord = 'fro'),2))
                while it < maxiter-1 and err >1e-6:
                    it  = it +1
                    T1 = 0.5*lambda2*np.dot(np.transpose(self.A),W2)
                    T1[T1<0] = 0
                    W1  = W1*np.dot(self.R,np.transpose(H1))/(eps+np.dot(W1,np.dot(H1,np.transpose(H1)))+0.5*mu*W1)
                    H1  = H1*(np.dot(np.transpose(W1),self.R))/(eps+np.dot(np.dot(np.transpose(W1),W1),H1))
                    T2 = 0.5*(lambda2/lambda1+eps)*np.dot(self.A,W1)
                    T2[T2<0] = 0
                    W2  = W2*(np.dot(self.G,np.transpose(H2))+T2)/(eps+np.dot(W2,np.dot(H2,np.transpose(H2)))+0.5*mu*W2)
                    H2  = H2*(np.dot(np.transpose(W2),self.G)/(eps+np.dot(np.dot(np.transpose(W2),W2),H2)))
                    m1  = np.zeros((self.N,self.N))
                    m2  = np.zeros((self.N,self.N))
                    for z in range(self.N):
                        m1[z,z] = LA.norm(H1[z,:])
                        m2[z,z] = LA.norm(H2[z,:])
                    
                    W2  = np.dot(W2,m2)
                    W1  = np.dot(W1,m1)
                    H1  = np.dot(LA.inv(m1),H1)
                    H2  = np.dot(LA.inv(m2),H2)
                    
                    terms[it] = lambda1*pow(LA.norm(self.G-np.dot(W2,H2),ord = 'fro'),2)+pow(LA.norm(self.R-np.dot(W1,H1),ord = 'fro'),2)+lambda2*pow(np.trace(np.dot(np.dot(np.transpose(W2),self.A),W1)),2)+mu*(pow(LA.norm(W1,ord = 'fro'),2)+pow(LA.norm(W2,ord = 'fro'),2))
                    err = abs(terms[it]-terms[it-1])/abs(terms[it-1])
        
                # latent embedding of cells H1 & H2
                S2=np.argmax(H2,0)
                S1=np.argmax(H1,0)

                p2 = np.zeros((self.G.shape[0],self.N))
                for i in range(self.N):
                    for j in range(self.G.shape[0]):
                        statistic, p2[j,i],df  = ttest_ind(self.G.iloc[j,S2==i], self.G.iloc[j,S2!=i] ,alternative='smaller')

                WP2 = np.zeros((W2.shape))
                p2[np.isnan(p2) ] = 1
                scores = -np.log10(p2)
                temp = int(len(E_symbol)/20)
                for i in range(self.N):
                    indexs = scores[:,i].argsort()[-temp:][::-1]
                    WP2[indexs,i] = 1
                    E_all[count,i,indexs] = 1
                    E_p_all[count,i,indexs] = p2[indexs,i]

                p1 = np.zeros((self.R.shape[0],self.N))
                for i in range(self.N):
                    for j in range(self.R.shape[0]):
                        statistic, p1[j,i],df  = ttest_ind(self.R.iloc[j,S1==i], self.R.iloc[j,S1!=i] ,alternative='smaller')

                WP1 = np.zeros((W1.shape))
                p1[np.isnan(p1) ] = 1
                scores = -np.log10(p1)
                temp = int(len(P_symbol)/20)
                for i in range(self.N):
                    indexs = scores[:,i].argsort()[-temp:][::-1]
                    WP1[indexs,i] = 1
                    P_all[count,i,indexs] = 1
                    P_p_all[count,i,indexs] = p1[indexs,i]

                    T = np.dot(np.dot(np.transpose(WP2),self.A),WP1)
                temp = np.sum(np.sum(T))*np.diag(1/np.sum(T,axis=0))*T*np.diag(1/np.sum(T,axis=1))
                detr1[x,y] = np.trace(temp)
                detr[x,y] = np.trace(T)
                S1_all[count] = S1
                S2_all[count] = S2
                count = count + 1
                    
        [i,j] = npmax(detr)
        print("Score is :"+ str(detr1[i,j]/self.N))
        print("If the score >=1, the clustering matching for scRNA-seq and scATAC-seq is well. Otherwise, we sugguest to tune the parameters.")

        index = detr.argmax()
        
        # cluster result for scATAC-Seq
        S1_final = S1_all[index,:]+1
        # cluster result for scRNA-Seq
        S2_final = S2_all[index,:]+1
        E_final  = E_all[index,:,:]
        P_final  = P_all[index,:,:]
        E_p_final  = E_p_all[index,:,:]
        P_p_final  = P_p_all[index,:,:]

        """
        fout1 = open("scATAC-result.txt","w")
        fout2 = open("scRNA-result.txt","w")
        fout3 = open("cluster-specific-peaks-genes-pairs.txt","w")

        print(S1_final)
        print(S2_final)

        for item in S1_final:
            fout1.write(str(item)+"\t")
        fout1.write("\n")


        for item in S2_final:
                fout2.write(str(item)+"\t")
        fout2.write("\n")

        for i in range(self.N):
            temp = np.dot(np.reshape(E_final[i,:],(self.G.shape[0],1)),np.reshape(P_final[i,:],(1,self.R.shape[0])))*self.A
            p, q = np.nonzero(temp)
            for j in range(len(p)):
                fout3.write("cluster "+str(i+1)+": "+E_symbol[p[j]]+"\t"+P_symbol[q[j]]+"\t"+str(E_p_final[i,p[j]])+"\t"+str(P_p_final[i,q[j]])+"\n")
        """

        return H1, H2