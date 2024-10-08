# Feature Selection Using Bayes Deep Learning

This project investigates the efficacy of **Bayes Deep Learning (BayesDL)** and the **TASSEL Mixed Linear Model (MLM)** in feature selection for whole-genome SNP data, focusing on their application to two datasets: the TASSEL test data and the Arabidopsis thaliana dataset. The primary objective is to identify significant **Single Nucleotide Polymorphisms (SNPs)** associated with phenotypic traits, comparing the performance of a deep learning approach against a traditional statistical method.

## Overview

### BayesDL:
- **Designed for high-dimensional data** and complex genetic architectures.
- **Strong capability** to detect complex, non-linear associations, particularly in the **Arabidopsis dataset**.
- Computationally intensive, requiring substantial resources, making **scaling for larger datasets** challenging.

### TASSEL MLM:
- Well-suited for handling **population structure** and **genetic relatedness**.
- Provides clear and interpretable results in simpler genetic environments.
- **Effective in initial SNP exploration**, especially in datasets with less genetic complexity.

## Key Findings

- **BayesDL** offers **superior performance** in complex scenarios but demands more computational resources.
- **TASSEL MLM** remains a valuable tool for **initial SNP exploration** and **simpler genetic environments**.
- A hybrid approach, integrating **BayesDL’s probabilistic strengths** with **MLM’s robust population control**, could enhance the accuracy and reliability of **genotype-phenotype association studies**.

## Conclusion

This comparative analysis suggests that both methods offer distinct advantages. Integrating **BayesDL** and **TASSEL MLM** can significantly advance genomic research and plant breeding strategies, providing comprehensive insights into **genetic architecture**.

## Future Work

Exploring **hybrid approaches** that combine the **non-linear capabilities** of BayesDL with the **interpretability and population structure control** of TASSEL MLM will likely yield even more powerful tools for **genomic prediction** and **association studies**.
