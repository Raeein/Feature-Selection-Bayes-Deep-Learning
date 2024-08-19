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
phenotype_name <- "FLC"

cat("\n BayesDL started for ",phenotype_name)

#Mention the number of top SNPs selected by the model.
snp_count <- 20

#Noting down the step execution
formatted_datetime <- function() {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
}

start_time <- Sys.time()

cat("\n Starting modelling",formatted_datetime())
#Continuous phenotype
#Import SNP (or feature) data set for 
######
xTrain <- read.csv(paste0("XTrain_", phenotype_name, "_prop.csv"))
xTrain <- xTrain[,-1]
xTrain <- as.data.frame(xTrain)
xTest <- read.csv(paste0("XTest_", phenotype_name, "_prop.csv"))
xTest <- xTest[,-1]
xTest <- as.data.frame(xTest)
######

#Import response (or phenotype) data set for
#Width_22
######
yTrain <- read.csv(paste0("ytrain_", phenotype_name, "_prop.csv"))
yTrain <- yTrain[,-1]
yTrain <- as.vector(yTrain)
yTest <- read.csv(paste0("ytest_", phenotype_name, "_prop.csv"))
yTest <- yTest[,-1]
yTest <- as.vector(yTest)

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

#Regression NN function
###### 
sm_reg <- stan_model("nn_reg.stan")

fit_nn_reg <- function(x_train, y_train, x_test, y_test, H, n_H, data, method = "optimize", ...) {
  stan_data <- list(
    N = nrow(x_train),
    P = ncol(x_train),
    x = x_train,
    y = y_train,
    H = H,
    n_H = n_H,
    N_test = length(y_test),
    y_test = y_test
  )
  if(method == "optimize") {
    optOut <- optimizing(sm_reg, data = stan_data)
    test_char <- paste0("output_test[", 1:stan_data$N_test, "]")
    y_test_pred <- optOut$par[test_char]
    mse <- mean((y_test_pred - y_test)^2)
    correlation <- cor(y_test_pred,y_test)
    rsq <- (correlation^2)
    out <- list(y_test_pred = y_test_pred,
                sigma = optOut$par["sigma"],
                mse  = mse,
                rsq = rsq,
                fit = optOut)
    return(out)
  } else {
    if(method == "sampling") {
      out <- sampling(sm_reg, data = stan_data, ...)
    } else if (method == "vb") {
      out <- vb(sm_reg, data = stan_data, pars = c("output_test", "sigma", "output_test_rng"), ...)
    }
    y_test_pred <- summary(out, pars = "output_test")$summary
    sigma <- summary(out, pars = "sigma")$summary
    out <- list(y_test_pred = y_test_pred,
                sigma = sigma,
                fit = out)
    return(out)
  }
}
######

#Fit regression NN
#Optimizing the model
fit_opt <- fit_nn_reg(xTest, yTest, xTrain, yTrain, 2, 50, data, method = "optimize")

#Calculate Performance Metrics
RMSE <- sqrt(fit_opt$mse)
R2 <- fit_opt$rsq

cat("\n RMSE : ",RMSE)
cat("\n R2 : ",R2)

cat("\n Sampling from the fitted model.", formatted_datetime())
#Sampling from the fitted model
fit_nuts <- fit_nn_reg(xTrain, yTrain, xTest, yTest, 2, 50, method = "sampling",
                       chains = 4, cores = 4, iter = 2000, warmup=1000)


#Save the fitted model for future use because running takes a while
saveRDS(fit_nuts, paste0("f_stan_fit_", phenotype_name, ".rds"))
cat("\n Model Saved successfully.")

# Reading the mode
fit <-readRDS(paste0("f_stan_fit_", phenotype_name, ".rds"))

#Find sampler for each chain
sample <- get_sampler_params(fit_nuts$fit, inc_warmup = TRUE)
lapply(sample, summary, digits = 2)
sapply(sample, FUN = colMeans)

#Parameter names of the draws
list_of_draws <- rstan::extract(fit_nuts$fit)
print(names(list_of_draws))

#Check the Rhat values
print(fit_nuts$fit)

#####
#Feature selection using BNN
cat("\n Feature selection using BNN", formatted_datetime())
######

#Extract weights associated with each predictor
cat("\n Extracting weights associated with each predictor.", formatted_datetime())
P = ncol(xTrain)
N = nrow(xTrain)
n_H = 50 #Number of nodes in each hidden layer
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
cat("\n Removing weight means equal to zero.", formatted_datetime())
lapply(wt_means, function(x) {x[x!=0]})

# Compute the variable importance measures
var_imp <- wt_sds/ abs(wt_means)

# Sort the variable importance measures in descending order
var_imp <- sort(var_imp, decreasing = FALSE)

# Select the top SNPs
cat("\n Selecting the top predictors.", formatted_datetime())
top_vars <- names(var_imp)[1:snp_count]
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

# Pairs Plot
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

# #Posterior Predictive checks
# params <- rstan::extract(fit_nuts$fit, pars = top_vars[1:2])
# 
# #Prior Distribution
# x <- rnorm(4000,mean = 0, sd = 1)
# x <- as.data.frame(x)
# x <- as.numeric(unlist(x))

#PPC Plot
# color_scheme_set("viridis")
# par(mfrow=c(1,2))
# ppc_dens_overlay((x), t(as.matrix(transformed_params$`data_to_hidden_weights[13,25]`)))
# ppc_dens_overlay((x), t(as.matrix(transformed_params$`data_to_hidden_weights[19,4]`)))

############

#Noting down the total runtime 
end_time <- Sys.time()

# Calculate the total run time
run_time <- end_time - start_time

# Print the start time, end time, and total run time
cat("\nStart time:", start_time, "\n")
cat("\nEnd time:", end_time, "\n")
cat("\nTotal run time:", run_time, "\n")
#############

