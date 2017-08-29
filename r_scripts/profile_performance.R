###############################################################################
# Author: lily-tian
# Last revision: 08/18/2017
###############################################################################

###############################################################################
# ENVIRONMENT SETUP
###############################################################################

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
# TRAINING VS TESTING SETS
###############################################################################

set.seed(100)
train <- sample(1:nrow(data), nrow(data)/2)
data_train <- data[train, ]
data_test <- data[!(1:nrow(data) %in% train), ]

###############################################################################
# LOGISTIC REGRESSION
###############################################################################

# define relevant variables
indep_vars <-c("tenure", "lnfavs", "hasprofile", "community")
dep_var <- c("isauthor")

# define regressand vector y and design matrix X
data_train <- na.omit(data_train[c(dep_var, indep_vars)])
y = data_train[[dep_var]]
X = as.matrix(data_train[indep_vars])

# find logit  
logit <- glm(y ~ X, family = "binomial")
summary(logit)

# define regressand vector y and design matrix X
data_test <- na.omit(data_test[c(dep_var, indep_vars)])
y = data_test[[dep_var]]
X = as.matrix(data_test[indep_vars])

# make prediction
guess <- predict(logit, newdata = data_test, type = "response")
guess[guess <= 0.5] = 0
guess[guess > 0.5] = 1

# calculate accuracy
sum(y == guess)/length(guess)

###############################################################################
# LINEAR REGRESSION
###############################################################################

# define relevant variables
indep_vars <-c("tenure", "lnfavs", "hasprofile", "community")
dep_var <- c("lnwritten")

# define regressand vector y and design matrix X
y = data_authors[[dep_var]]
X = as.matrix(data_authors[indep_vars])

# find linear regression
reg <- glm(y ~ X, family = "gaussian")
summary(reg)

# perform diagnostics


###############################################################################
###############################################################################
