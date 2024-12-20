#!/bin/bash
#SBATCH --nodes=1                      # Number of nodes to use
#SBATCH --job-name=TE_32_16
#SBATCH --output=z_%j.out
#SBATCH --error=z_%j.err
#SBATCH --time=10:00:00                # Maximum time for the job 
#SBATCH --mem=32G                      # Memory required per node 
#SBATCH --cpus-per-task=4              # Number of CPUs - don't need more than 4 
#SBATCH --mail-user=username@uoguelph.ca  
#SBATCH --mail-type=ALL                # Get email for all job events

# Load the R module
module load r/4.3  
#module load nixpkgs/16.09 

Rscript install-packages.R

# Run the R script
Rscript PreliminaryFeatureSelection_proposed.R
# Rscript PreliminaryFeatureSelection_ST_FRI.R
# Rscript Regression_BayesDL.R
Rscript Regression_BayesDL_FRI.R
# Rscript Classification_BayesDL.R
# Rscript Load_And_Plot_FRI.R
# Rscript Load_And_Plot.R
# Rscript Test_Scores.R