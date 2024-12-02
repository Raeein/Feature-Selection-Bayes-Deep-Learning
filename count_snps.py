# Script to count the number of valid SNPs available for each Phenotype. Specify phenotype_columns.
import pandas as pd

# Load the phenotype file
phenotype_file = "phenotypes.pheno"
phenotype_df = pd.read_csv(phenotype_file, delim_whitespace=True)

# List of phenotype columns to check. Below are phenotype for the Arabidopsis dataset
phenotype_columns = ["Silique_16", "Germ_22", "Width_22", "Emco5"]

valid_sample_counts = {}

# Iterate through each phenotype and count valid (non-NaN) values
for phenotype in phenotype_columns:
    valid_samples = phenotype_df[phenotype].notna().sum()
    valid_sample_counts[phenotype] = valid_samples

# Print the results
for phenotype, count in valid_sample_counts.items():
    print(f"{phenotype}: {count} valid samples")

# Save cleaned file (optional)
cleaned_df = phenotype_df.dropna(subset=phenotype_columns)
cleaned_df.to_csv("cleaned_phenotype_file.txt", sep="\t", index=False)
