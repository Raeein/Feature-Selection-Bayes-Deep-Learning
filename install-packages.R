
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    # For the digital alliance servers use this instead (4.4 is the R version I'm using):
    # install.packages(pkg, repos = "http://cran.us.r-project.org", lib="/home/<username>/R/x86_64-pc-linux-gnu-library/4.4")
    install.packages(pkg, repos = "http://cran.us.r-project.org")
  }
}

# List of CRAN packages
cran_packages <- c(
  "dplyr", "ROCR", "logr", "matrixStats", "glmnet", "SGL",
  "MLmetrics", "gglasso", "caTools", "caret", "tidyverse",
  "doParallel", "grpreg", "rstan", "magrittr", "pROC", "ggplot2",
  "bayesplot", "shinystan", "dendextend", "InformationValue", "foreach"
)

# Install CRAN packages
sapply(cran_packages, install_if_missing)

# Install Bioconductor packages
if (!requireNamespace("BiocManager", quietly = TRUE))
  # For the digital alliance servers use this instead (4.4 is the R version I'm using):
  # install.packages("BiocManager", repos = "http://cran.us.r-project.org", lib="/<username>/bagherir/R/x86_64-pc-linux-gnu-library/4.4")
  install.packages("BiocManager", repos = "http://cran.us.r-project.org")

BiocManager::install("trio")


# install_packages.R
# install.packages("dplyr", repos = "http://cran.us.r-project.org")
# install.packages("ROCR", repos = "http://cran.us.r-project.org")
# install.packages("logr", repos = "http://cran.us.r-project.org")
# install.packages("matrixStats", repos = "http://cran.us.r-project.org")
# install.packages("glmnet", repos = "http://cran.us.r-project.org")
# install.packages("SGL", repos = "http://cran.us.r-project.org")
# install.packages("MLmetrics", repos = "http://cran.us.r-project.org")
# install.packages("gglasso", repos = "http://cran.us.r-project.org")
# install.packages("caTools", repos = "http://cran.us.r-project.org")
# install.packages("caret", repos = "http://cran.us.r-project.org")
# install.packages("tidyverse", repos = "http://cran.us.r-project.org")
# install.packages("doParallel", repos = "http://cran.us.r-project.org")
# install.packages("grpreg", repos = "http://cran.us.r-project.org")
# BiocManager::install("trio")
# install.packages("rstan", repos = "http://cran.us.r-project.org")
# install.packages("magrittr", repos = "http://cran.us.r-project.org")
# install.packages("caret", repos = "http://cran.us.r-project.org")
# install.packages("pROC", repos = "http://cran.us.r-project.org")
# install.packages("ROCR", repos = "http://cran.us.r-project.org")
# install.packages("bayesplot", repos = "http://cran.us.r-project.org")
# install.packages("shinystan", repos = "http://cran.us.r-project.org")
# install.packages("dendextend", repos = "http://cran.us.r-project.org")
# install.packages("InformationValue", repos = "http://cran.us.r-project.org")
# install.packages("foreach", repos = "http://cran.us.r-project.org")
