# In[]
import scanpy as sc
import anndata
import numpy as np 
import pandas as pd
from scipy.sparse import save_npz, load_npz, csr_matrix
import MultiMAP
import warnings
warnings.filterwarnings("ignore")

sc.settings.set_figure_params(dpi=80)
# In[]
data = "simulated_protein2"
if data == "diag":
        #create the anndata for each batch of count matrix, should use raw count without highly variable genes, cells gene expression should only be log transformed
        data_dir = "./"
        rna = sc.read(data_dir + 'rna.h5ad')
        atac_peaks = sc.read(data_dir + 'atac-peaks.h5ad')
        atac_genes = sc.read(data_dir + 'atac-genes.h5ad')

        # calculate the reduced dimensional space of each dataset
        # for scATAC-Seq uses TF-IDF & LSI
        MultiMAP.TFIDF_LSI(atac_peaks)
        # note that the reduced space is saved into atac_genes instead of atac_peaks, as atac_genes is the only input
        atac_genes.obsm['X_lsi'] = atac_peaks.obsm['X_lsi'].copy()
        # calculate the pca space of rna
        rna_pca = rna.copy()
        sc.pp.scale(rna_pca)
        sc.pp.pca(rna_pca)
        rna.obsm['X_pca'] = rna_pca.obsm['X_pca'].copy()

        '''
        Integration(adatas, use_reps, scale=True, embedding=True, seed=0, **kwargs)
        Run MultiMAP to integrate a number of AnnData objects from various multi-omics experiments
        into a single joint dimensionally reduced space. Returns a joint object with the resulting 
        embedding stored in ``.obsm['X_multimap']`` (if instructed) and appropriate graphs in 
        ``.obsp``. The final object will be a concatenation of the individual ones provided on 
        input, so in the interest of ease of exploration it is recommended to have non-scaled data 
        in ``.X``.
        
        Input
        -----
        adatas : list of ``AnnData``
                The objects to integrate. The ``.var`` spaces will be intersected across subsets of 
                the objects to compute shared PCAs, so make sure that you have ample features in 
                common between the objects. ``.X`` data will be used for computation.
        use_reps : list of ``str``
                The ``.obsm`` fields for each of the corresponding ``adatas`` to use as the 
                dimensionality reduction to represent the full feature space of the object. Needs 
                to be precomputed and present in the object at the time of calling the function.
        scale : ``bool``, optional (default: ``True``)
                Whether to scale the data to N(0,1) on a per-dataset basis prior to computing the 
                cross-dataset PCAs. Improves integration.
        embedding : ``bool``, optional (default: ``True``)
                Whether to compute the MultiMAP embedding. If ``False``, will just return the graph,
                which can be used to compute a regular UMAP. This can produce a manifold quicker,
                but at the cost of accuracy.
        n_neighbors : ``int`` or ``None``, optional (default: ``None``)
                The number of neighbours for each node (data point) in the MultiGraph. If ``None``, 
                defaults to 15 times the number of input datasets.
        n_components : ``int`` (default: 2)
                The number of dimensions of the MultiMAP embedding.
        seed : ``int`` (default: 0)
                RNG seed.
        strengths: ``list`` of ``float`` or ``None`` (default: ``None``)
                The relative contribution of each dataset to the layout of the embedding. The 
                higher the strength the higher the weighting of its cross entropy in the layout loss. 
                If provided, needs to be a list with one 0-1 value per dataset; if ``None``, defaults 
                to 0.5 for each dataset.
        cardinality : ``float`` or ``None``, optional (default: ``None``)
                The target sum of the connectivities of each neighbourhood in the MultiGraph. If 
                ``None``, defaults to ``log2(n_neighbors)``.
        
        The following parameter definitions are sourced from UMAP 0.5.1:
        
        n_epochs : int (optional, default None)
                The number of training epochs to be used in optimizing the
                low dimensional embedding. Larger values result in more accurate
                embeddings. If None is specified a value will be selected based on
                the size of the input dataset (200 for large datasets, 500 for small).
        init : string (optional, default 'spectral')
                How to initialize the low dimensional embedding. Options are:
                        * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
                        * 'random': assign initial embedding positions at random.
                        * A numpy array of initial embedding positions.
        min_dist : float (optional, default 0.1)
                The effective minimum distance between embedded points. Smaller values
                will result in a more clustered/clumped embedding where nearby points
                on the manifold are drawn closer together, while larger values will
                result on a more even dispersal of points. The value should be set
                relative to the ``spread`` value, which determines the scale at which
                embedded points will be spread out.
        spread : float (optional, default 1.0)
                The effective scale of embedded points. In combination with ``min_dist``
                this determines how clustered/clumped the embedded points are.
        set_op_mix_ratio : float (optional, default 1.0)
                Interpolate between (fuzzy) union and intersection as the set operation
                used to combine local fuzzy simplicial sets to obtain a global fuzzy
                simplicial sets. Both fuzzy set operations use the product t-norm.
                The value of this parameter should be between 0.0 and 1.0; a value of
                1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
                intersection.
        local_connectivity : int (optional, default 1)
                The local connectivity required -- i.e. the number of nearest
                neighbors that should be assumed to be connected at a local level.
                The higher this value the more connected the manifold becomes
                locally. In practice this should be not more than the local intrinsic
                dimension of the manifold.
        a : float (optional, default None)
                More specific parameters controlling the embedding. If None these
                values are set automatically as determined by ``min_dist`` and
                ``spread``.
        b : float (optional, default None)
                More specific parameters controlling the embedding. If None these
                values are set automatically as determined by ``min_dist`` and
                ``spread``.
        '''
        adata = MultiMAP.Integration(adatas = [rna, atac_genes], use_reps = ['X_pca', 'X_lsi'])
        # adata.obsm["X_multimap"] is two dimensional, like umap visualization
        # adata.obsp["connectivities"] stores a cell by cell graph, note that the cells from all batches are included (seems to be a binarized graph).
        # adata.write_h5ad("outputs/multimap_int.h5ad")

        adata.obs[["source"]].to_csv("outputs/batch_id.csv")
        np.save("outputs/multimap.npy", adata.obsm["X_multimap"])
        save_npz("outputs/multimap_graph.npz", adata.obsp["connectivities"])

elif data == "pancreas":
        data_dir = "./"
        nbatches = 8
        rnas = []
        for batch in range(nbatches):
                rna = sc.read(data_dir + f"rna_{batch}.h5ad")
                rna_pca = rna.copy()
                sc.pp.scale(rna_pca)
                sc.pp.pca(rna_pca)
                rna.obsm["X_pca"] = rna_pca.obsm["X_pca"].copy()
                rnas.append(rna)

        adata = MultiMAP.Integration(adatas = rnas, use_reps = ["X_pca"] * len(rnas))
        adata.obs[["source"]].to_csv("outputs/batch_id.csv")
        np.save("outputs/multimap.npy", adata.obsm["X_multimap"])
        save_npz("outputs/multimap_graph.npz", adata.obsp["connectivities"])

elif data == "MOp":
        #create the anndata for each batch of count matrix, should use raw count without highly variable genes, cells gene expression should only be log transformed
        data_dir = "./"
        GxC1 = sc.read(data_dir + 'GxC1.h5ad')
        RxC1 = sc.read(data_dir + 'RxC1.h5ad')
        GxC2 = sc.read(data_dir + 'GxC2.h5ad')
        RxC3 = sc.read(data_dir + 'RxC3.h5ad')
        GxC3 = sc.read(data_dir + 'GxC3.h5ad')

        assert np.unique(GxC1.obs["source"].values) == "C1"
        assert np.unique(RxC1.obs["source"].values) == "C1"
        assert np.unique(GxC2.obs["source"].values) == "C2"
        assert np.unique(RxC3.obs["source"].values) == "C3"
        assert np.unique(GxC3.obs["source"].values) == "C3"

        # calculate the reduced dimensional space of each dataset
        # for scATAC-Seq uses TF-IDF & LSI
        MultiMAP.TFIDF_LSI(RxC3)
        # note that the reduced space is saved into atac_genes instead of atac_peaks, as atac_genes is the only input
        GxC3.obsm['X_lsi'] = RxC3.obsm['X_lsi'].copy()
        # for snare seq
        MultiMAP.TFIDF_LSI(RxC1)
        GxC1.obsm["X_lsi"] = RxC1.obsm["X_lsi"].copy()

        # calculate the pca space of rna
        rna_pca = GxC2.copy()
        sc.pp.scale(rna_pca)
        sc.pp.pca(rna_pca)
        GxC2.obsm['X_pca'] = rna_pca.obsm['X_pca'].copy()

        adata = MultiMAP.Integration(adatas = [GxC1, GxC2, GxC3], use_reps = ['X_lsi', 'X_pca', 'X_lsi'])
        # adata.obsm["X_multimap"] is two dimensional, like umap visualization
        # adata.obsp["connectivities"] stores a cell by cell graph, note that the cells from all batches are included (seems to be a binarized graph).
        
        adata.obs[["source"]].to_csv("outputs/batch_id.csv")
        np.save("outputs/multimap.npy", adata.obsm["X_multimap"])
        save_npz("outputs/multimap_graph.npz", adata.obsp["connectivities"])

elif data == "pbmc":
        data_dir = "./"
        GxC1 = sc.read(data_dir + 'adata_rna_c1.h5ad')
        PxC1 = sc.read(data_dir + 'adata_protein_c1.h5ad')
        GxC2 = sc.read(data_dir + 'adata_rna_c2.h5ad')
        PxC2 = sc.read(data_dir + 'adata_protein_c2.h5ad')
        PxC3 = sc.read(data_dir + 'adata_protein_c3.h5ad')
        RxC3 = sc.read(data_dir + 'adata_atac_c3.h5ad')
        PxC4 = sc.read(data_dir + 'adata_protein_c4.h5ad')
        RxC4 = sc.read(data_dir + 'adata_atac_c4.h5ad')
        
        counts_gxc1 = GxC1.X.todense()
        counts_pxc1 = PxC1.X
        counts_c1 = np.concatenate((counts_gxc1, counts_pxc1), axis = 1)
        counts_c1 = anndata.AnnData(X = csr_matrix(counts_c1), obs = PxC1.obs)
        counts_c1.var.index = np.concatenate((np.array(["gene_" + x for x in GxC1.var.index.values]), np.array(["protein_" + x for x in PxC1.var.index.values])), axis = 0)
        
        counts_gxc2 = GxC2.X.todense()
        counts_pxc2 = PxC2.X
        counts_c2 = np.concatenate((counts_gxc2, counts_pxc2), axis = 1)
        counts_c2 = anndata.AnnData(X = csr_matrix(counts_c2), obs = PxC2.obs)
        counts_c2.var.index = np.concatenate((np.array(["gene_" + x for x in GxC2.var.index.values]), np.array(["protein_" + x for x in PxC2.var.index.values])), axis = 0)
        
        counts_rxc3 = RxC3.X.todense()
        counts_pxc3 = PxC3.X
        counts_c3 = np.concatenate((counts_rxc3, counts_pxc3), axis = 1)
        counts_c3 = anndata.AnnData(X = csr_matrix(counts_c3), obs = PxC3.obs)
        counts_c3.var.index = np.concatenate((np.array(["region_" + x for x in RxC3.var.index.values]), np.array(["protein_" + x for x in PxC3.var.index.values])), axis = 0)
        
        counts_rxc4 = RxC4.X.todense()
        counts_pxc4 = PxC4.X
        counts_c4 = np.concatenate((counts_rxc4, counts_pxc4), axis = 1)
        counts_c4 = anndata.AnnData(X = csr_matrix(counts_c4), obs = PxC4.obs)
        counts_c4.var.index = np.concatenate((np.array(["region_" + x for x in RxC4.var.index.values]), np.array(["protein_" + x for x in PxC4.var.index.values])), axis = 0)        

        assert np.unique(GxC1.obs["source"].values) == "C1"
        assert np.unique(PxC1.obs["source"].values) == "C1"
        assert np.unique(GxC2.obs["source"].values) == "C2"
        assert np.unique(PxC2.obs["source"].values) == "C2"
        assert np.unique(RxC3.obs["source"].values) == "C3"
        assert np.unique(PxC3.obs["source"].values) == "C3"
        assert np.unique(RxC4.obs["source"].values) == "C4"
        assert np.unique(PxC4.obs["source"].values) == "C4"

        sc.pp.scale(counts_c1)
        sc.pp.pca(counts_c1)
        sc.pp.scale(counts_c2)
        sc.pp.pca(counts_c2)
        sc.pp.scale(counts_c3)
        sc.pp.pca(counts_c3)
        sc.pp.scale(counts_c4)
        sc.pp.pca(counts_c4)

        # NEW
        adata = MultiMAP.Integration(adatas = [counts_c1, counts_c2, counts_c3, counts_c4], use_reps = ['X_pca', 'X_pca', 'X_pca', 'X_pca'])
        adata.obs[["source"]].to_csv("outputs/batch_id.csv")
        np.save("outputs/multimap.npy", adata.obsm["X_multimap"])
        save_npz("outputs/multimap_graph.npz", adata.obsp["connectivities"])
        
        # OLD
        # # calculate the reduced dimensional space of each dataset
        # # for scATAC-Seq uses TF-IDF & LSI
        # MultiMAP.TFIDF_LSI(RxC3)
        # # note that the reduced space is saved into atac_genes instead of atac_peaks, as atac_genes is the only input
        # PxC3.obsm['X_lsi'] = RxC3.obsm['X_lsi'].copy()
        # # for snare seq
        # MultiMAP.TFIDF_LSI(RxC4)
        # PxC4.obsm["X_lsi"] = RxC4.obsm["X_lsi"].copy()

        # # calculate the pca space of rna
        # rna_pca = GxC2.copy()
        # sc.pp.scale(rna_pca)
        # sc.pp.pca(rna_pca)
        # PxC2.obsm['X_pca'] = rna_pca.obsm['X_pca'].copy()

        # rna_pca = GxC1.copy()
        # sc.pp.scale(rna_pca)
        # sc.pp.pca(rna_pca)
        # PxC1.obsm['X_pca'] = rna_pca.obsm['X_pca'].copy()


        # adata = MultiMAP.Integration(adatas = [PxC1, PxC2, PxC3, PxC4], use_reps = ['X_pca', 'X_pca', 'X_lsi', 'X_lsi'])
        # # adata.obsm["X_multimap"] is two dimensional, like umap visualization
        # # adata.obsp["connectivities"] stores a cell by cell graph, note that the cells from all batches are included (seems to be a binarized graph).
        
        # adata.obs[["source"]].to_csv("outputs/batch_id.csv")
        # np.save("outputs/multimap.npy", adata.obsm["X_multimap"])
        # save_npz("outputs/multimap_graph.npz", adata.obsp["connectivities"])

elif data == "simulated_scenario1":
        #create the anndata for each batch of count matrix, should use raw count without highly variable genes, cells gene expression should only be log transformed
        data_dir = "./"
        atac_genes1 = sc.read(data_dir + 'atac_genes_1.h5ad')
        atac_genes2 = sc.read(data_dir + 'atac_genes_2.h5ad')
        atac_genes3 = sc.read(data_dir + 'atac_genes_3.h5ad')
        atac_genes4 = sc.read(data_dir + 'atac_genes_4.h5ad')
        atac_genes5 = sc.read(data_dir + 'atac_genes_5.h5ad')
        atac_genes6 = sc.read(data_dir + 'atac_genes_6.h5ad')
        
        atac_peaks1 = sc.read(data_dir + 'atac_peaks_1.h5ad')
        atac_peaks2 = sc.read(data_dir + 'atac_peaks_2.h5ad')
        atac_peaks3 = sc.read(data_dir + 'atac_peaks_3.h5ad')
        atac_peaks4 = sc.read(data_dir + 'atac_peaks_4.h5ad')
        atac_peaks5 = sc.read(data_dir + 'atac_peaks_5.h5ad')
        atac_peaks6 = sc.read(data_dir + 'atac_peaks_6.h5ad')
        
        rna1 = sc.read(data_dir + 'rna_1.h5ad')
        rna2 = sc.read(data_dir + 'rna_2.h5ad')
        rna3 = sc.read(data_dir + 'rna_3.h5ad')
        rna4 = sc.read(data_dir + 'rna_4.h5ad')
        rna5 = sc.read(data_dir + 'rna_5.h5ad')
        rna6 = sc.read(data_dir + 'rna_6.h5ad')

        
        # calculate the pca space of rna
        rna_pca4 = rna4.copy()
        rna_pca5 = rna5.copy()
        rna_pca6 = rna6.copy()
        sc.pp.scale(rna_pca4)
        sc.pp.pca(rna_pca4)
        sc.pp.scale(rna_pca5)
        sc.pp.pca(rna_pca5)
        sc.pp.scale(rna_pca6)
        sc.pp.pca(rna_pca6)
        rna4.obsm['X_pca'] = rna_pca4.obsm['X_pca'].copy()
        rna5.obsm['X_pca'] = rna_pca5.obsm['X_pca'].copy()
        rna6.obsm['X_pca'] = rna_pca6.obsm['X_pca'].copy()

        

        # calculate the reduced dimensional space of each dataset
        # for scATAC-Seq uses TF-IDF & LSI
        MultiMAP.TFIDF_LSI(atac_peaks1)
        MultiMAP.TFIDF_LSI(atac_peaks2)
        MultiMAP.TFIDF_LSI(atac_peaks3)
        # print(np.max(atac_peaks1.obsm['X_lsi']))
        # print(np.max(atac_peaks2.obsm['X_lsi']))
        # print(np.max(atac_peaks3.obsm['X_lsi']))

        # note that the reduced space is saved into atac_genes instead of atac_peaks, as atac_genes is the only input
        atac_genes1.obsm['X_lsi'] = atac_peaks1.obsm['X_lsi'].copy()
        atac_genes2.obsm['X_lsi'] = atac_peaks2.obsm['X_lsi'].copy()
        atac_genes3.obsm['X_lsi'] = atac_peaks3.obsm['X_lsi'].copy()

        adata = MultiMAP.Integration(adatas = [atac_genes1, atac_genes2, atac_genes3, rna4, rna5, rna6], use_reps = ['X_lsi', 'X_lsi', 'X_lsi', 'X_pca', 'X_pca', 'X_pca'])
        # adata.obsm["X_multimap"] is two dimensional, like umap visualization
        # adata.obsp["connectivities"] stores a cell by cell graph, note that the cells from all batches are included (seems to be a binarized graph).
        # adata.write_h5ad("outputs/multimap_int.h5ad")

        adata.obs[["source"]].to_csv("outputs/batch_id.csv")
        np.save("outputs/multimap.npy", adata.obsm["X_multimap"])
        save_npz("outputs/multimap_graph.npz", adata.obsp["connectivities"])

elif data == "simulated_scenario2":
        #create the anndata for each batch of count matrix, should use raw count without highly variable genes, cells gene expression should only be log transformed
        data_dir = "./"
        atac_genes1 = sc.read(data_dir + 'atac_genes_1.h5ad')
        atac_genes2 = sc.read(data_dir + 'atac_genes_2.h5ad')
        atac_genes3 = sc.read(data_dir + 'atac_genes_3.h5ad')
        atac_genes4 = sc.read(data_dir + 'atac_genes_4.h5ad')
        atac_genes5 = sc.read(data_dir + 'atac_genes_5.h5ad')
        atac_genes6 = sc.read(data_dir + 'atac_genes_6.h5ad')
        
        atac_peaks1 = sc.read(data_dir + 'atac_peaks_1.h5ad')
        atac_peaks2 = sc.read(data_dir + 'atac_peaks_2.h5ad')
        atac_peaks3 = sc.read(data_dir + 'atac_peaks_3.h5ad')
        atac_peaks4 = sc.read(data_dir + 'atac_peaks_4.h5ad')
        atac_peaks5 = sc.read(data_dir + 'atac_peaks_5.h5ad')
        atac_peaks6 = sc.read(data_dir + 'atac_peaks_6.h5ad')
        
        rna1 = sc.read(data_dir + 'rna_1.h5ad')
        rna2 = sc.read(data_dir + 'rna_2.h5ad')
        rna3 = sc.read(data_dir + 'rna_3.h5ad')
        rna4 = sc.read(data_dir + 'rna_4.h5ad')
        rna5 = sc.read(data_dir + 'rna_5.h5ad')
        rna6 = sc.read(data_dir + 'rna_6.h5ad')


        paired4 = anndata.AnnData(X = csr_matrix(np.concatenate((atac_peaks4.X.todense(), atac_genes4.X.todense()), axis = 1)))
        paired4.var.index = np.concatenate((atac_peaks4.var.index.values.squeeze(), rna4.var.index.values.squeeze()), axis = 0)
        paired4.obs = atac_peaks4.obs

        # calculate the pca space of rna
        paired4_pca4 = paired4.copy()
        rna_pca5 = rna5.copy()
        rna_pca6 = rna6.copy()
        # there can be genes with the same value across dataset, scale make them nan
        sc.pp.scale(paired4_pca4)
        paired4_pca4.X = np.where(np.isnan(paired4_pca4.X), 0, paired4_pca4.X)
        sc.pp.pca(paired4_pca4)
        sc.pp.scale(rna_pca5)
        sc.pp.pca(rna_pca5)
        sc.pp.scale(rna_pca6)
        sc.pp.pca(rna_pca6)
        paired4.obsm['X_pca'] = paired4_pca4.obsm['X_pca'].copy()
        rna5.obsm['X_pca'] = rna_pca5.obsm['X_pca'].copy()
        rna6.obsm['X_pca'] = rna_pca6.obsm['X_pca'].copy()



        # calculate the reduced dimensional space of each dataset
        # for scATAC-Seq uses TF-IDF & LSI
        MultiMAP.TFIDF_LSI(atac_peaks1)
        MultiMAP.TFIDF_LSI(atac_peaks2)
        MultiMAP.TFIDF_LSI(atac_peaks3)
        # print(np.max(atac_peaks1.obsm['X_lsi']))
        # print(np.max(atac_peaks2.obsm['X_lsi']))
        # print(np.max(atac_peaks3.obsm['X_lsi']))

        # note that the reduced space is saved into atac_genes instead of atac_peaks, as atac_genes is the only input
        atac_genes1.obsm['X_lsi'] = atac_peaks1.obsm['X_lsi'].copy()
        atac_genes2.obsm['X_lsi'] = atac_peaks2.obsm['X_lsi'].copy()
        atac_genes3.obsm['X_lsi'] = atac_peaks3.obsm['X_lsi'].copy()
        adatas = [atac_genes1, atac_genes2, atac_genes3, paired4, rna5, rna6]
        for i in range(len(adatas)):
                sc.pp.scale(adatas[i])
                adatas[i].X = np.where(np.isnan(adatas[i].X), 0, adatas[i].X)
        adata = MultiMAP.Integration(adatas = adatas, use_reps = ['X_lsi', 'X_lsi', 'X_lsi', 'X_pca', 'X_pca', 'X_pca'], scale = False)
        # adata.obsm["X_multimap"] is two dimensional, like umap visualization
        # adata.obsp["connectivities"] stores a cell by cell graph, note that the cells from all batches are included (seems to be a binarized graph).
        # adata.write_h5ad("outputs/multimap_int.h5ad")

        adata.obs[["source"]].to_csv("outputs/batch_id.csv")
        np.save("outputs/multimap.npy", adata.obsm["X_multimap"])
        save_npz("outputs/multimap_graph.npz", adata.obsp["connectivities"])

elif data == "simulated_protein1":
        #create the anndata for each batch of count matrix, should use raw count without highly variable genes, cells gene expression should only be log transformed
        data_dir = "./"
        atac_proteins1 = sc.read(data_dir + 'atac_proteins_1.h5ad')
        atac_proteins2 = sc.read(data_dir + 'atac_proteins_2.h5ad')
        atac_proteins3 = sc.read(data_dir + 'atac_proteins_3.h5ad')
        atac_proteins4 = sc.read(data_dir + 'atac_proteins_4.h5ad')
        atac_proteins5 = sc.read(data_dir + 'atac_proteins_5.h5ad')
        atac_proteins6 = sc.read(data_dir + 'atac_proteins_6.h5ad')
        
        atac_peaks1 = sc.read(data_dir + 'atac_peaks_1.h5ad')
        atac_peaks2 = sc.read(data_dir + 'atac_peaks_2.h5ad')
        atac_peaks3 = sc.read(data_dir + 'atac_peaks_3.h5ad')
        atac_peaks4 = sc.read(data_dir + 'atac_peaks_4.h5ad')
        atac_peaks5 = sc.read(data_dir + 'atac_peaks_5.h5ad')
        atac_peaks6 = sc.read(data_dir + 'atac_peaks_6.h5ad')
        
        rna_genes1 = sc.read(data_dir + 'rna_1.h5ad')
        rna_genes2 = sc.read(data_dir + 'rna_2.h5ad')
        rna_genes3 = sc.read(data_dir + 'rna_3.h5ad')
        rna_genes4 = sc.read(data_dir + 'rna_4.h5ad')
        rna_genes5 = sc.read(data_dir + 'rna_5.h5ad')
        rna_genes6 = sc.read(data_dir + 'rna_6.h5ad')

        rna_proteins1 = sc.read(data_dir + 'rna_proteins_1.h5ad')
        rna_proteins2 = sc.read(data_dir + 'rna_proteins_2.h5ad')
        rna_proteins3 = sc.read(data_dir + 'rna_proteins_3.h5ad')
        rna_proteins4 = sc.read(data_dir + 'rna_proteins_4.h5ad')
        rna_proteins5 = sc.read(data_dir + 'rna_proteins_5.h5ad')
        rna_proteins6 = sc.read(data_dir + 'rna_proteins_6.h5ad')

        protein1 = sc.read(data_dir + 'proteins_1.h5ad')
        protein2 = sc.read(data_dir + 'proteins_2.h5ad')
        protein3 = sc.read(data_dir + 'proteins_3.h5ad')
        protein4 = sc.read(data_dir + 'proteins_4.h5ad')
        protein5 = sc.read(data_dir + 'proteins_5.h5ad')
        protein6 = sc.read(data_dir + 'proteins_6.h5ad')

         # calculate the pca space of rna
        protein_pca3 = protein3.copy()
        protein_pca4 = protein4.copy()
        sc.pp.scale(protein_pca3)
        sc.pp.pca(protein_pca3)
        sc.pp.scale(protein_pca4)
        sc.pp.pca(protein_pca4)
        protein3.obsm['X_pca'] = protein_pca3.obsm['X_pca'].copy()
        protein4.obsm['X_pca'] = protein_pca4.obsm['X_pca'].copy()

        # calculate the reduced dimensional space of each dataset
        # for scATAC-Seq uses TF-IDF & LSI
        MultiMAP.TFIDF_LSI(atac_peaks1)
        MultiMAP.TFIDF_LSI(atac_peaks2)

        sc.pp.scale(rna_genes5)
        sc.pp.pca(rna_genes5)
        sc.pp.scale(rna_genes6)
        sc.pp.pca(rna_genes6)

        # note that the reduced space is saved into atac_genes instead of atac_peaks, as atac_genes is the only input
        atac_proteins1.obsm['X_lsi'] = atac_peaks1.obsm['X_lsi'].copy()
        atac_proteins2.obsm['X_lsi'] = atac_peaks2.obsm['X_lsi'].copy()
        rna_proteins5.obsm['X_pca'] = rna_genes5.obsm['X_pca'].copy()
        rna_proteins6.obsm['X_pca'] = rna_genes6.obsm['X_pca'].copy()

        adatas = [atac_proteins1, atac_proteins2, protein3, protein4, rna_proteins5, rna_proteins6]
        for i in range(len(adatas)):
                sc.pp.scale(adatas[i])
                adatas[i].X = np.where(np.isnan(adatas[i].X), 0, adatas[i].X)
        adata = MultiMAP.Integration(adatas = adatas, use_reps = ['X_lsi', 'X_lsi', 'X_pca', 'X_pca', 'X_pca', 'X_pca'], scale = False)

        # adata.obsm["X_multimap"] is two dimensional, like umap visualization
        # adata.obsp["connectivities"] stores a cell by cell graph, note that the cells from all batches are included (seems to be a binarized graph).
        # adata.write_h5ad("outputs/multimap_int.h5ad")

        adata.obs[["source"]].to_csv("outputs/batch_id.csv")
        np.save("outputs/multimap.npy", adata.obsm["X_multimap"])
        save_npz("outputs/multimap_graph.npz", adata.obsp["connectivities"])

elif data == "simulated_protein2":
        #create the anndata for each batch of count matrix, should use raw count without highly variable genes, cells gene expression should only be log transformed
        data_dir = "./"
        atac_proteins1 = sc.read(data_dir + 'atac_proteins_1.h5ad')
        atac_proteins2 = sc.read(data_dir + 'atac_proteins_2.h5ad')
        atac_proteins3 = sc.read(data_dir + 'atac_proteins_3.h5ad')
        atac_proteins4 = sc.read(data_dir + 'atac_proteins_4.h5ad')
        atac_proteins5 = sc.read(data_dir + 'atac_proteins_5.h5ad')
        atac_proteins6 = sc.read(data_dir + 'atac_proteins_6.h5ad')
        
        atac_peaks1 = sc.read(data_dir + 'atac_peaks_1.h5ad')
        atac_peaks2 = sc.read(data_dir + 'atac_peaks_2.h5ad')
        atac_peaks3 = sc.read(data_dir + 'atac_peaks_3.h5ad')
        atac_peaks4 = sc.read(data_dir + 'atac_peaks_4.h5ad')
        atac_peaks5 = sc.read(data_dir + 'atac_peaks_5.h5ad')
        atac_peaks6 = sc.read(data_dir + 'atac_peaks_6.h5ad')
        
        rna_genes1 = sc.read(data_dir + 'rna_1.h5ad')
        rna_genes2 = sc.read(data_dir + 'rna_2.h5ad')
        rna_genes3 = sc.read(data_dir + 'rna_3.h5ad')
        rna_genes4 = sc.read(data_dir + 'rna_4.h5ad')
        rna_genes5 = sc.read(data_dir + 'rna_5.h5ad')
        rna_genes6 = sc.read(data_dir + 'rna_6.h5ad')

        rna_proteins1 = sc.read(data_dir + 'rna_proteins_1.h5ad')
        rna_proteins2 = sc.read(data_dir + 'rna_proteins_2.h5ad')
        rna_proteins3 = sc.read(data_dir + 'rna_proteins_3.h5ad')
        rna_proteins4 = sc.read(data_dir + 'rna_proteins_4.h5ad')
        rna_proteins5 = sc.read(data_dir + 'rna_proteins_5.h5ad')
        rna_proteins6 = sc.read(data_dir + 'rna_proteins_6.h5ad')

        protein1 = sc.read(data_dir + 'proteins_1.h5ad')
        protein2 = sc.read(data_dir + 'proteins_2.h5ad')
        protein3 = sc.read(data_dir + 'proteins_3.h5ad')
        protein4 = sc.read(data_dir + 'proteins_4.h5ad')
        protein5 = sc.read(data_dir + 'proteins_5.h5ad')
        protein6 = sc.read(data_dir + 'proteins_6.h5ad')

        paired3 = anndata.AnnData(X = csr_matrix(np.concatenate((atac_peaks3.X.todense(), protein3.X.todense()), axis = 1)))
        paired3.var.index = np.concatenate((atac_peaks3.var.index.values.squeeze(), protein3.var.index.values.squeeze()), axis = 0)
        paired3.obs = atac_peaks3.obs

        paired4 = anndata.AnnData(X = csr_matrix(np.concatenate((rna_genes4.X.todense(), protein4.X.todense()), axis = 1)))
        paired4.var.index = np.concatenate((rna_genes4.var.index.values.squeeze(), protein4.var.index.values.squeeze()), axis = 0)
        paired4.obs = rna_genes4.obs

         # calculate the pca space of rna
        paired_pca3 = paired3.copy()
        paired_pca4 = paired4.copy()
        sc.pp.scale(paired_pca3)
        paired_pca3.X = np.where(np.isnan(paired_pca3.X), 0, paired_pca3.X)
        sc.pp.pca(paired_pca3)
        sc.pp.scale(paired_pca4)
        paired_pca4.X = np.where(np.isnan(paired_pca4.X), 0, paired_pca4.X)
        sc.pp.pca(paired_pca4)
        paired3.obsm['X_pca'] = paired_pca3.obsm['X_pca'].copy()
        paired4.obsm['X_pca'] = paired_pca4.obsm['X_pca'].copy()

        # calculate the reduced dimensional space of each dataset
        # for scATAC-Seq uses TF-IDF & LSI
        MultiMAP.TFIDF_LSI(atac_peaks1)
        MultiMAP.TFIDF_LSI(atac_peaks2)

        sc.pp.scale(rna_genes5)
        rna_genes5.X = np.where(np.isnan(rna_genes5.X), 0, rna_genes5.X)
        sc.pp.pca(rna_genes5)
        sc.pp.scale(rna_genes6)
        rna_genes6.X = np.where(np.isnan(rna_genes6.X), 0, rna_genes6.X)
        sc.pp.pca(rna_genes6)

        # note that the reduced space is saved into atac_genes instead of atac_peaks, as atac_genes is the only input
        atac_proteins1.obsm['X_lsi'] = atac_peaks1.obsm['X_lsi'].copy()
        atac_proteins2.obsm['X_lsi'] = atac_peaks2.obsm['X_lsi'].copy()
        rna_proteins5.obsm['X_pca'] = rna_genes5.obsm['X_pca'].copy()
        rna_proteins6.obsm['X_pca'] = rna_genes6.obsm['X_pca'].copy()

        adatas = [atac_proteins1, atac_proteins2, paired3, paired4, rna_proteins5, rna_proteins6]
        for i in range(len(adatas)):
                sc.pp.scale(adatas[i])
                adatas[i].X = np.where(np.isnan(adatas[i].X), 0, adatas[i].X)
        adata = MultiMAP.Integration(adatas = adatas, use_reps = ['X_lsi', 'X_lsi', 'X_pca', 'X_pca', 'X_pca', 'X_pca'], scale = False)
        # adata.obsm["X_multimap"] is two dimensional, like umap visualization
        # adata.obsp["connectivities"] stores a cell by cell graph, note that the cells from all batches are included (seems to be a binarized graph).
        # adata.write_h5ad("outputs/multimap_int.h5ad")

        adata.obs[["source"]].to_csv("outputs/batch_id.csv")
        np.save("outputs/multimap.npy", adata.obsm["X_multimap"])
        save_npz("outputs/multimap_graph.npz", adata.obsp["connectivities"])

# %%
