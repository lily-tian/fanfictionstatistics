###############################################################################
# Author: lily-tian
# Last revision: 08/18/2017
###############################################################################

###############################################################################
# ENVIRONMENT SETUP
###############################################################################

# clears working evironment
rm(list = ls())

###############################################################################
# DATA SETUP
###############################################################################

# retrieves data
data <- read.csv("./../data/clean_data/df_profile.csv")

# reformats data
data["author"] <- 0
data[data$isauthor == "True", "author"] <- 1
data["profile"] <- 0
data[data$hasprofile == "True", "profile"] <- 1
data["favs"] <- data["fa"] + data["fs"]

# subsets data
df_active <- subset(data, status != 'inactive')

###############################################################################
# LOGISTIC REGRESSION
###############################################################################

# defines relevant variables
indep_vars <-c("cc", "favs", "tenure", "profile")
dep_var <- c("author")

# define regressand vector y and design matrix X
y = df_active[[dep_var]]
X = as.matrix(df_active[indep_vars])
  
logit <- glm(y ~ X, family = "binomial")
summary(logit)