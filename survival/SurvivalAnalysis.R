library(survival)
library(penalized)

# create command line arguements
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0){stop("No path was provided for the survival data")}

# get data
df <- read.csv(paste(args[1], '/survival_data.csv', sep=''), header=TRUE)
SurvivalObject <- with(df, Surv(Time, Status == "Dead")) # make first column a Survival Object
DataMatrix <- df
DataMatrix[, 2] <- SurvivalObject
DataMatrix <- DataMatrix[, -1]
                      
# stepwise penalised Cox Regression
L1_stepwise_regression  <- penalized(formula(x = DataMatrix), data = DataMatrix, lambda2 = 10, step=40)

# plot coefficients
png(paste(args[1], '/plots/survival_plot.png', sep=''), width=600, 600)
plotpath(L1_stepwise_regression, main="Coefficients for Stepwise Penalized Cox Regression", labelsize = 1.0, standardize = TRUE, cex.main=1.75)
dev.off()