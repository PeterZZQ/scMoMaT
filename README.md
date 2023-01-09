## scMoMaT: a unified framework for single cell mosaic integration and multi-modal bio-marker detection
scMoMaT v0.1.0


#### Description
scMoMaT is a single-cell data integration method that is able to:
* integrate single cell multi-omics data under the mosaic scenario using matrix tri-factorization
* uncover the cell type specific bio-markers at the same time when learning a unified cell representation
* integrate cell batches with unequal cell type composition

<img src = "fig1.png" width = 700ptx>

#### Dependency
```
Python >= 3.8.10
numpy >= 1.19.5
torch >= 1.11.0
scipy >= 1.7.3
sklearn >= 0.22.2.post1
matplotlib >= 3.5.2
umap >= 0.5.2
adjustText >= 0.7.3 (optional)
pandas >= 1.4.0 (optional)
```


#### Directory
* `src` contains the main script of scMoMaT
* `test` contains the testing script of scMoMaT on the datasets in the manuscript and running script of baseline methods. Please refer to this folder for the reproduction of test results. The testing dataset is available upon request.
* `data` stores the datasets

#### Data
Necessary data for PBMC (the first real dataset in the manuscript, `ASAP-PBMC`) and MOp (the second real dataset in the manuscript, `MOp_5batches`) are provided, which should be suffice for the running of scMoMaT in `demo_pbmc.py` and `demo_mop_5batches.py` as examples. The dataset for the other demo scrips are available upon requests. 

#### Installation and usage 
No installation is needed.

`demo_scmomat.ipynb` provides a example run on `MOp_5batches` dataset (running time: training procedure takes ~5mins on NVIDIA A40 GPU, total file takes ~30mins). For more example, please see the test scripts in `test` folder, necessary comments are included:
* `test_pbmc.py`: The first real dataset in the manuscript (**data provided**).
* `test_bmmc_healthy.py`: The third real dataset in the maunscript.
* `test_spleen.py`: The fourth real dataset in the manuscript.
* `test_spleen_subsample.py`: The fourth real dataset in the manuscript.
* `test_mop_5batches.py`: The second real dataset in the manuscript.


#### Contact
* Ziqi Zhang: ziqi.zhang@gatech.edu
* Xiuwei Zhang: xiuwei.zhang@gatech.edu 


#### Cite
```
@article{zhang2022scmomat,
  title={scMoMaT: Mosaic integration of single cell multi-omics matrices using matrix trifactorization},
  author={Zhang, Ziqi and Sun, Haoran and Chen, Xinyu and Mariappan, Ragunathan and Chen, Xi and Jain, Mika and Efremova, Mirjana and Rajan, Vaibhav and Teichmann, Sarah and Zhang, Xiuwei},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory},
}
```