rm(list = ls())
gc()
# install.packages(topGo)
# install.packages(org.Hs.eg.db)
# https://jdblischak.github.io/nw/analysis/mouse/go.html
library(topGO)
# library(org.Mm.eg.db)
library(org.Hs.eg.db)

# First create the gene universe. 
# This is all the genes tested for differential expression assigned a 1 for differentially expressed and 0 if not.

path = "./marker_gene_candidates/"
background_gene <- read.csv('../../data/hori/Pancreas/gene.csv', sep = ",", header = FALSE)[[1]]
gene_candidates <- read.csv(paste0(path,"cluster0genes.csv"), sep = ",", header = TRUE)[[1]]
gene_universe <- as.integer(background_gene%in%gene_candidates)
gene_universe <- factor(gene_universe)
names(gene_universe) <- background_gene

# Create the topGO data object. 
# Only consider “Biological Process” categories(which is mostly the case) and use the Mouse Ensembl database for annotation.
go_data <- new("topGOdata",
               ontology = "BP",
               allGenes = gene_universe,
               nodeSize = 5,
               annotationFun = annFUN.org,
               mapping = "org.Hs.eg.db",
               ID = "symbol")

# performing enrichment test
# Use the classic algorithm and score the tests with Fisher’s exact test.

# https://bioconductor.org/packages/release/bioc/vignettes/topGO/inst/doc/topGO.pdf see all algorithm
# default is weight01
go_test <- runTest(go_data, algorithm = "weight01", statistic = "fisher")

# analysis of the result
# Keep the results with a Fisher’s exact test p-value < 0.05.
go_table <- GenTable(go_data, weightFisher = go_test,
                     orderBy = "weightFisher", ranksOf = "weightFisher",
                     topNodes = sum(score(go_test) < .05),
                     numChar=1000)
head(go_table)

write.table(go_table, file = paste0(path, "GO_marker_genes_1.csv"), sep = ",")
