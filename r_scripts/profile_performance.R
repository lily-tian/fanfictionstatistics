###############################################################################
# Author: lily-tian
# Last revision: 08/18/2017
###############################################################################

###############################################################################
# ENVIRONMENT SETUP
###############################################################################

# import libraries
library(stargazer)

# clear working evironment
rm(list = ls())

###############################################################################
# DATA SETUP
###############################################################################

# retrieve raw data
data_raw <- read.csv("./../data/clean_data/df_profile.csv")

# create working copy
data <- subset(data_raw, status != 'inactive')

###############################################################################
# VARIABLE SETUP
###############################################################################

# list all variables 
all_vars <- c("isauthor", "lnwritten", "tenure", "lnfavs", "hasprofile", 
             "community")

# create variables
data["community"] = (data["cc"] > 0) + 0

# redefine variables
data["favs"] <- data["fa"] + data["fs"]

# transform variables
data["lnfavs"] <- log(data["favs"] + 1)
data["lnwritten"] <- log(data["st"])

# subset data
data <- data[, all_vars]
data_authors <- subset(data, isauthor == 1)

###############################################################################
# LOGISTIC REGRESSION
###############################################################################

# define relevant variables
indep_vars <-c("tenure", "lnfavs", "hasprofile", "community")
dep_var <- c("isauthor")

# define regressand vector y and design matrix X
data_reg <- na.omit(data[c(dep_var, indep_vars)])
y = data_reg[[dep_var]]
X = as.matrix(data_reg[indep_vars])

# find logit  
logit <- glm(y ~ X, family = "binomial")
summary(logit)

# make prediction
guess <- predict(logit, type = "response")
guess[guess <= 0.5] = 0
guess[guess > 0.5] = 1

# calculate accuracy
1 - sum(abs(y - guess))/length(guess)

###############################################################################
# LINEAR REGRESSION
###############################################################################

# defines relevant variables
indep_vars <-c("cc", "lnfavs", "tenure", "profile")
dep_var <- c("lnwritten")

# define regressand vector y and design matrix X
y = data_authors[[dep_var]]
X = as.matrix(data_authors[indep_vars])

reg <- glm(y ~ X, family = "gaussian")
summary(reg)
