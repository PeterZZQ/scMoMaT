## scMoMaT: Mosaic integration of single cell multi-omics matrices using matrix trifactorization
scMoMaT is a single-cell data integration method that is able to:
* integrate single cell multi-omics data under the mosaic scenario using matrix tri-factorization
* uncover the cell type specific bio-markers at the same time when learning a unified cell representation
* integrate cell batches with unequal cell type composition

#### Directory
* `src` contains the main script of scMoMaT
* `test` contains the testing script of scMoMaT on the datasets in the manuscript.

#### Usage 
See the test scripts in `test` folder:
* `demo_pbmc.py`
* `demo_bmmc_healthy.py`
* `demo_spleen_subsample.py`
* `demo_mop.py`
