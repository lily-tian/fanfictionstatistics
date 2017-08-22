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

###############################################################################
# VARIABLE SETUP
###############################################################################

# create working copy
data <- subset(data_raw, status != 'inactive')

# create variables
data["author"] = (data["status"] == "author") + 0
data["community"] = (data["cc"] > 0) + 0
data["profile"] = data["profile"]

# redefine variables
data["favs"] <- data["fa"] + data["fs"]

# transform variables
data["lnfavs"] <- log(data["favs"] + 1)
data["lnwritten"] <- log(data["st"])

# subset data
data_authors <- subset(data, status == "author")

###############################################################################
# LOGISTIC REGRESSION
###############################################################################

# define relevant variables
indep_vars <-c("community", "lnfavs", "tenure", "profile")
dep_var <- c("author")

# define regressand vector y and design matrix X
data_reg <- na.omit(data[c(dep_var, indep_vars)])
y = data_reg[[dep_var]]
X = as.matrix(data_reg[indep_vars])

# find logit  
logit <- glm(y ~ X, family = "binomial")
summary(logit)

ourguess <- predict(logit, type = "response")
ourguess[ourguess <= 0.5] = 0
ourguess[ourguess > 0.5] = 1

1 - sum(abs(y - ourguess))/length(ourguess)

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
