# Clear environment
##############
rm(list=ls())
gc()
options(warn=-1)

#####
library(rstan)
library(magrittr)
library(caret)
library(tidyverse)
library(dplyr)
library(pROC)
library(ROCR)
library(bayesplot)
library(shinystan)
library(ggplot2)
######

#Mention the phenotype that need to be analysed
phenotype_name <- "Anthocyanin_22"

cat("\n BayesDL started for ",phenotype_name)


#Noting down the step execution
formatted_datetime <- function() {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
}

#Binary phenotype
#Import SNP (or feature) data set for 
#Anthocyanin_22
#####
cat("\n Xtrain_antho read : ",formatted_datetime())
xTrain <- read.csv(paste0("XTrain_", phenotype_name, ".csv"))
xTrain <- xTrain[,-1]
xTrain <- as.data.frame(xTrain)

cat("\n Xtest_antho read : ",formatted_datetime())
xTest <- read.csv(paste0("XTest_", phenotype_name, ".csv"))
xTest <- xTest[,-1]
xTest <- as.data.frame(xTest)
#####

#Import response (or phenotype) data set for
#Anthocyanin_22
#####
cat("\n Ytrain_antho read : ",formatted_datetime())
yTrain <- read.csv(paste0("ytrain_", phenotype_name, ".csv"))
yTrain <- yTrain[,-1]
yTrain <- as.vector(yTrain)
yTrain <- replace(yTrain, yTrain==1, 2)
yTrain <- replace(yTrain, yTrain==0, 1)

cat("\n Ytrain_antho read : ",formatted_datetime())
yTest <- read.csv(paste0("ytest_", phenotype_name, ".csv"))
yTest <- yTest[,-1]
yTest <- as.vector(yTest)
yTest <- replace(yTest, yTest==1, 2)
yTest <- replace(yTest, yTest==0, 1)

######

# Combine the training and testing datasets for the predictors (features) into one data frame
x <- bind_rows(xTrain, xTest)

# Combine the training and testing datasets for the response variable into one data frame
y <- as.data.frame(c(yTrain, yTest))

# Rename the column of the response variable data frame to "y"
colnames(y) <- "y"

# Bind the predictors (features) data frame and the response variable data frame into one data frame
data <- bind_cols(x, y)


######
#Bayesian Neural Network (BNN)
#Classification NN function 
######

# Load the Stan model from the file "nn_class.stan"
sm <- stan_model("nn_class.stan")

# Define a function to fit a neural network classifier using Stan
fit_nn_cat <- function(x_train, y_train, x_test, y_test, H, n_H, data, method = "optimizing", ...) {
  
  # Create a list of data to be passed to the Stan model
  stan_data <- list(
    N = nrow(x_train),         # Number of training samples
    P = ncol(x_train),         # Number of predictors (features)
    x = x_train,               # Training data for predictors
    labels = y_train,          # Training data for response variable
    H = H,                     # Number of hidden layers
    n_H = n_H,                 # Number of units in each hidden layer
    N_test = length(y_test)    # Number of test samples
  )
  
  # Check if the method is "optimizing"
  if(method == "optimizing") {
    # Use the optimizing method from Stan to fit the model
    optOut <- optimizing(sm, data = stan_data)
    
    # Create a character vector to extract the test predictions from the optimized output
    test_char <- paste0("output_test[", 1:length(y_test), ",", rep(1:max(y_train), each = length(y_test)), "]")
    
    # Reshape the predicted test values into a matrix
    y_test_pred <- matrix(optOut$par[test_char], stan_data$N_test, max(y_train))
    
    # Determine the predicted category by finding the maximum probability for each test sample
    y_test_cat <- apply(y_test_pred, 1, which.max)
    
    # Create a list containing the predicted values, predicted categories, confusion matrix, and optimization output
    out <- list(y_test_pred = y_test_pred,
                y_test_cat = y_test_cat,
                conf = table(y_test_cat, y_test),
                fit = optOut)
    
    # Return the output list
    return(out)
    
    # Check if the method is "sampling"
  } else if(method == "sampling") {
    # Use the sampling method from Stan to fit the model
    out <- sampling(sm, data = stan_data, ...)
    
    # Summarize the predicted test values from the sampling output
    y_test_pred <- summary(out, pars = "output_test")$summary
    
    # Create a list containing the predicted values and sampling output
    out <- list(y_test_pred = y_test_pred,
                fit = out)
    
    # Return the output list
    return(out)
  }
}

######

#Fit class NN
#Optimizing the model

# Fit the neural network classifier using the optimizing method
fit_opt <- fit_nn_cat(xTest, yTest, xTrain, yTrain, 2, 50, data, method = "optimizing")

# Calculate performance metrics

# Extract the confusion matrix from the fitted model
cm <- fit_opt$conf

# Compute accuracy
# # Accuracy is the proportion of true results (both true positives and true negatives) among the total number of cases examined
# accuracy <- sum(cm[1], cm[4]) / sum(cm[1:4])
# print(paste("Accuracy:", accuracy))
# 
# # Precision is the proportion of true positives among the total number of positive predictions
# precision <- cm[4] / sum(cm[4], cm[2])
# print(paste("Precision:", precision))
# 
# # Compute sensitivity (recall)
# # Sensitivity is the proportion of true positives among the total number of actual positives
# sensitivity <- cm[4] / sum(cm[4], cm[3])
# print(paste("Sensitivity:", sensitivity))
# 
# # Compute F1 score
# # F1 score is the harmonic mean of precision and sensitivity
# fscore <- (2 * (sensitivity * precision)) / (sensitivity + precision)
# print(paste("F1 Score:", fscore))
# 
# Compute AUC (Area Under the Curve)
# AUC is a performance metric for binary classification problems at various threshold settings
# auc <- auc(as.matrix(yTest), fit_opt$y_test_cat)
# print(paste("AUC:", auc))

cat("\n Sampling from the fitted model.", formatted_datetime())
#Sampling from the fitted model
fit_nuts <- fit_nn_cat(xTrain, yTrain, xTest, yTest, 2, 50, method = "sampling", 
                       chains = 4, cores = 4, iter = 2000, warmup=1000)

#Save the fitted model for future use because running takes a while
saveRDS(fit_nuts,paste0("stan_fit_", phenotype_name, ".rds"))
cat("\n Model Saved successfully.")

# Reading the saved Stan model from the file "stan_fit_antho.rds"
fit <- readRDS(paste0("stan_fit_", phenotype_name, ".rds"))

# Find sampler parameters for each chain, including warm-up iterations
sample <- get_sampler_params(fit_nuts$fit, inc_warmup = TRUE)

# Summarize the sampler parameters for each chain, rounding the results to 2 decimal places
lapply(sample, summary, digits = 2)

# Compute the column means of the sampler parameters for each chain
sapply(sample, FUN = colMeans)

# Extract parameter names from the draws of the Stan fit object
list_of_draws <- rstan::extract(fit_nuts$fit)
print(names(list_of_draws))

# Check Rhat values to assess convergence of the chains
print(fit_nuts$fit)


######
#Feature selection using BNN
cat("\n Feature selection using BNN", formatted_datetime())
######

#Extract weights associated with each predictor (or SNP)
cat("\n Extracting weights associated with each predictor.", formatted_datetime())
P = ncol(xTrain)
N = nrow(xTrain)
n_H = 50
wt_samples <- matrix(NA, nrow = P, ncol = n_H)

# Define the names of the parameters which are to be extracted
cat("\n Defining the name of the parameters", formatted_datetime())
param_names <- c()
for (i in 1:P){
  for(j in 1:n_H){
    wt_name <- paste0("data_to_hidden_weights[", i, ",", j, "]")
    param_names <- c(param_names, wt_name)
  }
}

# Extract the parameter samples and store them in a matrix
cat("\n Extracting the parameter using BNN", formatted_datetime())
wt_samples <- matrix(NA, nrow = 4000, ncol = length(param_names))
col_name <- list()
for (i in 1:length(param_names)) {
  wt_name <- param_names[i]
  wt_samples[, i] <- rstan::extract(fit_nuts$fit, pars = wt_name)[[1]]
  wt_samples <- as.data.frame(wt_samples)
  col_name = append(col_name,param_names[i])
}
colnames(wt_samples) <- col_name

# Compute the posterior means and standard deviations for each predictor (or SNP)
cat("\n Computing the posterior means and standard deviations for each predictor.", formatted_datetime())
wt_means <- colMeans(wt_samples)
wt_sds <- apply(wt_samples, 2, sd)

#Remove weight means that are equal to zero
cat("Removing weight means equal to zero.", formatted_datetime())
lapply(wt_means, function(x) {x[x!=0]})

# Compute the variable importance measures
var_imp <- wt_sds/ abs(wt_means) 

# Sort the variable importance measures in ascending order
var_imp <- sort(var_imp, decreasing = FALSE)

# Select the top 10 predictors
cat("\n Selecting the top 20 predictors.", formatted_datetime())
top_vars <- names(var_imp)[1:20]
df <- as.data.frame(top_vars)

# Extract the first number before the comma from each row
df$first_num <- str_extract(as.character(df$top_vars), "\\d+")
df$top_vars
num <- as.integer(df$first_num)
selectedSNPs <- list()
for (i in num) {
  selectedSNPs <- append(selectedSNPs, colnames(xTrain)[i])
  print(colnames(xTrain)[i])
}
selectedSNPs <- as.data.frame(selectedSNPs)
write.csv(df, file = paste0(phenotype_name, "_selected_weights_df_prop.csv"))
write.csv(selectedSNPs, file = paste0(phenotype_name, "_selected_snp_prop.csv"))


######

#MCMC Diagnosis for top 2 SNPs
#######
#Nuts and posterior parameters
np_draws <- nuts_params(fit_nuts$fit)
posterior_draws <- as.array(fit_nuts$fit)

#########
# Pairs Plot
cat("\n Save Mcmc pairs_plot")
color_scheme_set("red")
# Generate the mcmc_pairs plot and assign it to an object
mcmc_pairs_plot <- mcmc_pairs(posterior_draws, np = np_draws, pars = top_vars[1:2],
                              off_diag_args = list(size = 0.75),
                              main = top_vars[1:2])

# Save the plot to a file
ggsave(filename = paste0(phenotype_name, "_mcmc_pairs_plot.png"), plot = mcmc_pairs_plot, width = 10, height = 8, dpi = 300)


#Trace Plot
cat("\n Save Trace plot")
color_scheme_set("green")
mcmc_trace_plot <- mcmc_trace(posterior_draws, pars = top_vars[1:2], np = np_draws) +
  xlab("Post-warmup iteration")

# Save the plot to a file
ggsave(filename = paste0(phenotype_name, "_mcmc_trace_plot.png"), plot = mcmc_trace_plot, width = 10, height = 8, dpi = 300)


#Acf plot
cat("\n Save MCMC ACF Plot")
color_scheme_set("mix-brightblue-gray")
mcmc_acf_plot <- mcmc_acf(posterior_draws, pars = top_vars[1:2], lags = 35)
# Save the plot to a file
ggsave(filename = paste0(phenotype_name, "_mcmc_acf_plot.png"), plot = mcmc_acf_plot, width = 10, height = 8, dpi = 300)

#Posterior Predictive checks
params <- rstan::extract(fit_nuts$fit, pars = top_vars[1:2])

#Prior Distribution
# x <- rnorm(4000,mean = 0, sd = 1)
# x <- as.data.frame(x)
# x <- as.numeric(unlist(x))
# 
# #PPC Plot
# color_scheme_set("viridis")
# par(mfrow=c(1,2))
# ppc_dens_overlay((x), t(as.matrix(transformed_params$`data_to_hidden_weights[89,5]`)))
# ppc_dens_overlay((x), t(as.matrix(transformed_params$`data_to_hidden_weights[15,23]`)))
