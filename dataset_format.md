# Dataset Formatting Guide for Running the Code

To run the code with new datasets, you will need to prepare and format three files. These files must follow specific formatting guidelines to be able to work with the program. The required files are:

- **genotype.ped**
- **genotype.map**
- **phenotypes.pheno**
  

All three files should be space-separated and placed in the root of the project directory. Additionally, the file names should match exactly with those provided here to ensure the code can locate them and read their contents.

## File Descriptions and Formatting

### 1. phenotypes.pheno

This file contains the phenotype information for each individual in the study. It should include at least three columns:

**Format:**

- FID: Family ID
- IID: Individual ID
- Phenotypic Data: The third column should represent the primary phenotypic trait of interest (such as EarHT).
  
#### Example Format:

| FID   | IID   | EarHT |
|-------|-------|-------|
| 33-16 | 33-16 | 64.75 |
| 38-11 | 38-11 | 92.25 |
| 4226  | 4226  | 65.50 |
| 4722  | 4722  | 81.13 |
| A188  | A188  | 27.50 |


### 2. genotype.ped

The genotype.ped file contains the genotype information for each individual in the study. The file format is space-separated. The first six columns represent individual and family information, and the remaining columns contain genotype data (two columns per SNP for the two alleles). Note that this file format does not actually have columns and it is one long string of text.

**Format:**

- Column 1: Individual ID (IID)
- Column 2: Paternal ID
- Column 3: Maternal ID
- Column 4: Sex (1 = male, 2 = female, 0 or -9 = unknown)
- Column 5: Phenotype (usually -9 for missing phenotype data)
- Columns 6 onward: Genotype data (two columns per SNP, one for each allele)

#### Example Format:
```plaintext
33-16 -9 -9 -9 -9 C C C C G G T T G G T T G G T T C C C C T T T T C C ...
Nov-38 -9 -9 -9 -9 C C G G A A C C G G T T T T C C C C G G C C G G ...
4226 -9 -9 -9 -9 A A G G T T C C A A G G T T G G C C C C T T T T C C ...
```

### 3. genotype.map

The genotype.map file contains the genotype information for each individual in the study. It includes family, individual, parental IDs, sex, and genotype data for multiple genetic markers.

**Format:**

- First column (Chromosome): The chromosome number where the SNP is located.
- Second column (SNP ID): The identifier for each SNP.
- Third column (Genetic Distance): The genetic distance between the SNPs.
- Fourth column (Position): The base-pair position of the SNP on the chromosome.

#### Example Format:
| Chromosome | SNP ID | Genetic Distance | Position |
|------------|--------|------------------|----------|
| 1          | SNP1   | 0                | 123456   |
| 1          | SNP2   | 0                | 234567   |
| 2          | SNP3   | 0                | 345678   |
| 2          | SNP4   | 0                | 456789   |



## Summary of Formatting Requirements:

- All three files should be space-separated.
- phenotypes.pheno should include FID, IID, and the relevant phenotypic data.
- The genotype.ped file should have matching FID and IID columns with phenotypes.pheno.
- The first value in genotype.ped should match the first value of FID in phenotypes.pheno.
- All files should be placed in the root of the project directory.
- For additional information on the file formats, check out the official **plink** documentation at https://zzz.bwh.harvard.edu/plink/data.shtml