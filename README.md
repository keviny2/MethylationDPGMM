# MethylationDPGMM
##Multi-modal Pan-Cancer Stratification Using DNA Methylation Data
####*Kevin Yang*
###Abstract
**Motivation**: DNA methylation is an epigenetic mark that is suspected to have regulatory roles in a broad
range of biological processes and diseases. There is evidence that DNA methylation levels are associated
with transcriptional repression which may be closely related to cancer. Technology is now available to
reliably measure DNA methylation in large samples and genome-wide, allowing for large scale analysis
of methylation data. A common objective is to identify a latent structure shared across cancers from
different tissue types reflecting commonly altered gene pathways. These latent structures stratify cancer
patients into functionally similar groups. This process of stratification is commonly done using gene
expression data; now that DNA methylation data is available in large quantities, this study investigates
using both data types in stratification. We first perform important preprocessing steps on raw methylation
data; specifically, normal approximations are used to transform methylation values and feature selection
methods are applied to extract the most informative genes. Then we perform cluster analysis using a
Dirichlet Process Mixture Model.

**Results**: Our study suggests that DNA methylation provides key information for stratification. Cluster
analysis on DNA methylation data independently shows latent structures among cancer patients of
differing types. In particular, we discover a shared latent structure among ovarian, colorectal and breast
cancer patients. For joint analysis, two types of RNA expression data are used: RNA-seq and RNA-array.
We combine both types with DNA methylation and perform a cluster analysis. When DNA methylation
and RNA-seq data are combined, lung and brain cancer patients cluster together; whereas, when DNA
methylation and RNA-array data are combined, brain and blood cancer patients cluster together. These
results motivate further study on using DNA methylation as an additional feature to RNA expression.