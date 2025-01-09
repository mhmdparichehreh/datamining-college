##Required libraries-----------------------------
library(readr)            #import .CSV files
library("corrplot")       #Visualization of Correlation Matrix
library("moments")        #Moments, skewness, kurtosis and related tests
library("MASS")           #Box-Cox Transformations for Linear Models
library("leaps")          #Regression Subset Selection
library("glmnet")         #Lasso and Elastic-Net Regularized GLM
library("rpart")          #Classification and Regression Trees
library("rpart.plot")     #Plot Decision Tree
library("randomForest")   #Random Forests for Classification and Regression

##Read Data from File----------------------------
#college dataset : Statistics for a large number of US Colleges from the 1995 issue of US News and World Report

college <- read.csv("College.csv", header = TRUE)
data1 <- college
View(data1)

##Understanding the Business Question------------
#university Analytics
##Data inspection--------------------------------
dim(data1) #777*19
str(data1)
names(data1)

#A data frame with 777 observations on the following 18 variables.
#Private: A factor with levels No and Yes indicating private or public university
#Apps: Number of applications received
#Accept: Number of applications accepted
#Enroll: Number of new students enrolled
#Top10perc: Pct. new students from top 10% of H.S. class
#Top25perc: Pct. new students from top 25% of H.S. class
#F.Undergrad: Number of fulltime undergraduates
#P.Undergrad: Number of parttime undergraduates
#Outstate: Out-of-state tuition
#Room.Board: Room and board costs
#Books: Estimated book costs
#Personal: Estimated personal spending
#PhD: Pct. of faculty with Ph.D.’s
#Terminal: Pct. of faculty with terminal degree
#S.F.Ratio: Student/faculty ratio
#perc.alumni: Pct. alumni who donate
#Expend: Instructional expenditure per student
#Grad.Rate: Graduation rate

summary(data1)

#Convert categorical variables to factor
cat_var <- "Private"
data1[, cat_var] <- lapply(data1[, cat_var], factor)

summary(data1)
head(data1)
dim(data1)

#Dealing w/ MVs
#Analysis of MVs should be done:
sum(is.na(data1))
#fortunately we have no Na's cell here <3

#Remove "university Names" and "Private" column: 
data2 <- data1[,-c(1,2)]

head(data2)
dim(data2) #777*17
sum(is.na(data2))
summary(data2)

#Continuous variables distribution
par(mar = c(2, 2, 2, 2))
par(mfrow = c(3, 6))  # 3 rows and 6 columns
for (i in c(1:17)) {
  hist(data2[,i], xlab = "", main = paste("Hist. of", names(data2)[i]))
}

par(mfrow = c(1, 1))
boxplot(data2$Apps, main = "Apps Dist.")

#Identify outliers 
tukey_ul <- quantile(data2$Apps, probs = 0.75) + 1.5 * IQR(data2$Apps)
tukey_ul
sum(data2$Apps > tukey_ul)
# 70/777= 0.09% of total data

#Correlation Analysis
cor_table <- round(cor(data2[, c(1:17)]), 2)
View(cor_table)
corrplot(cor_table)

#Scatter Plot
par(mar = c(2, 2, 2, 2))
par(mfrow = c(3, 6))  # 3 rows and 6 columns
for (i in c(1:17)) {
  plot(data2[,i], data2$Apps, xlab = "", main = paste("Apps vs.", names(data2)[i]))
}

#Divide Dataset into Train and Test--------------
set.seed(1234)
train_cases <- sample(1:nrow(data2), nrow(data2) * 0.8)
train <- data2[train_cases,]
test  <- data2[- train_cases,]

dim(train)
summary(train)
dim(test)
summary(test)

#Train dataset w/o outliers
trimmed_train <- train[- which(train$Apps > tukey_ul), ]
dim(trimmed_train)
summary(trimmed_train$Apps)

##Building Prediction Model----------------------
#Model1 : Traditional Linear Regression----------
lm_1 <- lm(Apps ~ ., data = train)
summary(lm_1)

lm_2 <- lm(Apps ~ Enroll + Top10perc + Top25perc + Outstate + Expend + Grad.Rate , data = train)
summary(lm_2)

#Check Assumptions of Regression
#Normality of residuals
hist(lm_2$residuals, probability = TRUE, breaks = 25)
lines(density(lm_2$residuals), col = "red")

#QQ-plot
qqnorm(lm_2$residuals, main = "QQ Plot of residuals", pch = 20)
qqline(lm_2$residuals, col = "red")

#Test for Skewness and Kurtosis
#Good for sample size > 25
#Jarque-Bera Test (Skewness = 0 ?)
#p-value < 0.05 reject normality assumption
jarque.test(lm_2$residuals)

#Anscombe-Glynn Test (Kurtosis = 3 ?)
#p-value < 0.05 reject normality assumption
anscombe.test(lm_2$residuals)

#Note: Residuals are not Normally Distributed!

#Diagnostic Plots
plot(lm_2)

#Check multicollinearity
car :: vif(lm_1)
car :: vif(lm_2)

#Conclusion: severe violation of regression assumption
#Bad model!
#Test the Model----------------------------------
#Model: m1_1
#Prediction
pred_lm <- predict(lm_2, test)

#Absolute error mean, median, sd, max, min-------
abs_err_lm <- abs(pred_lm - test$Apps)
mean(abs_err_lm)
median(abs_err_lm)
sd(abs_err_lm)
range(abs_err_lm)

#histogram and boxplot
hist(abs_err_lm, breaks = 15)
boxplot(abs_err_lm)

#Actual vs. Predicted
plot(test$Apps, pred_lm, xlab = "Actual", ylab = "Prediction")
abline(a = 0, b = 1, col = "red", lwd = 2)

#Model2 : Box-Cox Transformation-----------------
#Box-Cox Transformation
box_results <- boxcox(Apps ~ ., data = train, lambda = seq(-5, 5, 0.1))
box_results <- data.frame(box_results$x, box_results$y)            # Create a data frame with the results
lambda <- box_results[which(box_results$box_results.y == max(box_results$box_results.y)), 1]
lambda

#log transformation
train$Log_Apps <- log(train$Apps)

#Model w/ Log Apps
lm_logresp_1 <- lm(Log_Apps ~ . - Apps, data = train)
summary(lm_logresp_1)

lm_logresp_2 <- lm(Log_Apps ~ Enroll + S.F.Ratio + perc.alumni + Expend + Grad.Rate, data = train)
summary(lm_logresp_2)

#Check Assumptions of Regression
plot(lm_logresp_1)
car :: vif(lm_logresp_1)

#Two Problems with Model2:
#   Multicolinearity
#   Not use of so many variables

#Model3: Using the Best Subset Selection Methods----------
#Algorithm:
# 1- Let M0 denote the null model , which contains no predictors. 
# 2- For k = 1, 2,...p:
# (a) Fit all Cr(p, k) models that contain exactly k predictors.
# (b) Pick the best among these models, and call it Mk.
#       The best is defined as having the largest R-squared.
# 3- Select a single best model from among M0, ..., Mp
#   using cross-validated prediction error, Cp, BIC, or adjusted R-squared
#Best Subset Selection---------------------------
bestsub_1 <- regsubsets(Log_Apps ~ . - Apps, nvmax = 16, data = train, method = "exhaustive")
summary(bestsub_1)

#Model Selection
#R-squared
summary(bestsub_1)$rsq

#Adjusted R-squared
#AdjR2 = 1 - [(1 - R2)(1 - n)/(n - d - 1)]
# n: the number of samples 
# d: the number of predictors

#Plot Adjusted R-squared
plot(summary(bestsub_1)$adjr2, type="b", xlab="# of Variables", ylab="AdjR2", xaxt='n', xlim=c(1, 16)); grid()
axis(1, at=1:16, labels=1:16)

points(which.max(summary(bestsub_1)$adjr2), summary(bestsub_1)$adjr2[which.max(summary(bestsub_1)$adjr2)], col = "red", cex = 2, pch = 20)

#Cp
#Cp = 1/n * (RSS + 2 * d * sigma_hat ^ 2)
# n: the number of samples 
# RSS: Residual Sum of Squares
# d: the number of predictors
# sigma_hat: estimate of the variance of the error (estimated on a model containing all predictors) 

#Plot Cp
plot(summary(bestsub_1)$cp, type = "b", xlab = "# of Variables", ylab = "Cp", xaxt = 'n', xlim = c(1, 16)); grid()
axis(1, at = 1:16, labels = 1:16)

points(which.min(summary(bestsub_1)$cp), summary(bestsub_1)$cp[which.min(summary(bestsub_1)$cp)], col = "red", cex = 2, pch = 20)

#BIC
#BIC (Bayesian Information Criterion ) =  -2 * LogLikelihood  + log(n) * d
# n: the number of samples 
# RSS: Residual Sum of Squares
# d: the number of predictors
# sigma_hat: estimate of the variance of the error 

#Plot BIC
plot(summary(bestsub_1)$bic, type = "b", xlab = "# of Variables", ylab = "BIC", xaxt = 'n', xlim = c(1, 16)); grid()
axis(1, at = 1:16, labels = 1:16)

points(which.min(summary(bestsub_1)$bic), summary(bestsub_1)$bic[which.min(summary(bestsub_1)$bic)], col = "red", cex = 2, pch = 20)

#Coefficients of the best model
coef(bestsub_1, 9) #Model w/ 9 variables

bestsub_2 <- lm(Log_Apps ~ Accept + Enroll + Top25perc + Room.Board + PhD + S.F.Ratio + perc.alumni + Expend + Grad.Rate, data = train)
summary(bestsub_2)

#Test the Model----------------------------------
#Prediction
pred_bestsub  <- predict(bestsub_2, test)
pred_bestsub 
pred_bestsub  <- exp(pred_bestsub)
pred_bestsub 
#Absolute error mean, median, sd, max, min-------
abs_err_bestsub <- abs(pred_bestsub - test$Apps)
mean(abs_err_bestsub)
median(abs_err_bestsub)
sd(abs_err_bestsub)
range(abs_err_bestsub)

#histogram and boxplot
hist(abs_err_bestsub, breaks = 15)
boxplot(abs_err_bestsub)

#Actual vs. Predicted
plot(test$Apps, pred_bestsub, xlab = "Actual", ylab = "Prediction")
abline(a = 0, b = 1, col = "red", lwd = 2)

#Forward and Backward Stepwise Selection---------
#Forward Selection
#Algorithm:
# 1- Let M0 denote the null model , which contains no predictors. 
# 2- For k = 0, 2,...p - 1:
# (a) Consider all p ??? k models that augment the predictors in Mk 
#       with one additional predictor.
# (b) Pick the best among these models, and call it Mk+1.
#       The best is defined as having the largest R-squared.
#3- Select a single best model from among M0, ..., Mp
#   using cross-validated prediction error, Cp, BIC, or adjusted R

fwd_1 <- regsubsets(Log_Apps ~ . - Apps, nvmax = 16, data = train, method = "forward")
summary(fwd_1)

which.max(summary(fwd_1)$adjr2)
which.min(summary(fwd_1)$cp)
which.min(summary(fwd_1)$bic)

#backward Selection
#Algorithm:
# 1- Let Mp denote the full model , which contains all predictors. 
# 2- For k = p, p - 1,..., 1:
# (a) Consider all k models that contain all but one of the predictors
#       in  Mk, for a total of k ??? 1 predictors.
# (b) Pick the best among these models, and call it Mk-1.
#       The best is defined as having the largest R-squared.
#3- Select a single best model from among M0, ..., Mp
#   using cross-validated prediction error, Cp, BIC, or adjusted R

bwd_1 <- regsubsets(Log_Apps ~ . - Apps, nvmax = 16, data = train, method = "backward")
summary(bwd_1)

which.max(summary(bwd_1)$adjr2)
which.min(summary(bwd_1)$cp)
which.min(summary(bwd_1)$bic)

coef(bestsub_1, 9)
coef(fwd_1, 9)
coef(bwd_1, 9)

#Model4: Using K-fold Cross-Validation Approach---------
k <- 10
set.seed(123)
folds <- sample(1:k, nrow(train), rep = TRUE)
cv_errors <- matrix(NA, k, 16, dimnames = list(NULL , paste(1:16)))
cv_errors

#Create prediction function for regsubsets()
#Prediction function

predict_regsubsets <- function(object, newdata, id) {
  reg_formula <- as.formula(object$call[[2]])
  mat <- model.matrix(reg_formula, newdata)
  coef_i <- coef(object, id = id)
  mat[, names(coef_i)] %*% coef_i
}

#K-fold Cross Validation
set.seed(1234)
for(i in 1:k){
  best_fit <- regsubsets(Log_Apps ~ . - Apps, data = train[folds != i,], nvmax = 16, method = "exhaustive")
  for(j in 1:16){
    pred <- predict_regsubsets(best_fit, newdata = train[folds == i,], id = j)
    cv_errors[i, j] <- mean((train$Log_Apps[folds == i] - pred) ^ 2)
  }
}

View(cv_errors)
mean_cv_errors <- apply(cv_errors, 2, mean)
mean_cv_errors 
plot(mean_cv_errors, type = "b")
which.min(mean_cv_errors)

#Coefficients of the best model
coef(bestsub_1, 16) #Model w/ 16 variables

bestsub_cv_1 <- lm(Log_Apps ~ Accept + Enroll + Top10perc + Top25perc + F.Undergrad + P.Undergrad + Outstate + Room.Board + Books + Personal + PhD + Terminal + S.F.Ratio + perc.alumni + Expend + Grad.Rate , data = train)
summary(bestsub_cv_1)

#Test the Model----------------------------------
#Prediction
pred_bestsub_cv <- predict(bestsub_cv_1, test)
pred_bestsub_cv <- exp(pred_bestsub_cv)
summary(pred_bestsub_cv)

#Absolute error mean, median, sd, max, min-------
abs_err_bestsub_cv <- abs(pred_bestsub_cv - test$Apps)
mean(abs_err_bestsub_cv)
median(abs_err_bestsub_cv)
sd(abs_err_bestsub_cv)
range(abs_err_bestsub_cv)

#histogram and boxplot
hist(abs_err_bestsub_cv, breaks = 15)
boxplot(abs_err_bestsub_cv)

#Actual vs. Predicted
plot(test$Apps, pred_bestsub_cv, xlab = "Actual", ylab = "Prediction")
abline(a = 0, b = 1, col = "red", lwd = 2)

#Model 5: Best Sub Selection Using Trimmed Train and CV--------------
#Add Log Salary
trimmed_train$Log_Apps <- log(trimmed_train$Apps)

trimmed_bestsub_1 <- regsubsets(Log_Apps ~ . - Apps, nvmax = 16, data = trimmed_train, method = "exhaustive")
summary(trimmed_bestsub_1)

which.max(summary(trimmed_bestsub_1)$adjr2)
which.min(summary(trimmed_bestsub_1)$cp)
which.min(summary(trimmed_bestsub_1)$bic)

#Coefficients of the best model
coef(trimmed_bestsub_1, 6) #Model w/ 10 variables

trimmed_bestsub_2 <- lm(Log_Apps ~ Accept + Top10perc + Terminal + S.F.Ratio + Expend + Grad.Rate, data = trimmed_train)
summary(trimmed_bestsub_2)

#Test the Model----------------------------------
#Prediction
pred_trimmed_bestsub  <- predict(trimmed_bestsub_2, test)
pred_trimmed_bestsub  <- exp(pred_trimmed_bestsub)
pred_trimmed_bestsub

#Absolute error mean, median, sd, max, min-------
abs_err_trimmed_bestsub <- abs(pred_trimmed_bestsub - test$Apps)
mean(abs_err_trimmed_bestsub)
median(abs_err_trimmed_bestsub)
sd(abs_err_trimmed_bestsub)
range(abs_err_trimmed_bestsub)

#histogram and boxplot
hist(abs_err_trimmed_bestsub, breaks = 15)
boxplot(abs_err_trimmed_bestsub)

#Actual vs. Predicted
plot(test$Apps, pred_trimmed_bestsub, xlab = "Actual", ylab = "Prediction")
abline(a = 0, b = 1, col = "red", lwd = 2)

#Comparisons of Models--------------------------
df <- data.frame("Model_1" = abs_err_lm, 
                 "Model_3" = abs_err_bestsub, 
                 "Model_4" = abs_err_bestsub_cv,
                 "Model_5" = abs_err_trimmed_bestsub)

models_comp <- data.frame("Mean of AbsErrors"   = apply(df, 2, mean),
                          "Median of AbsErrors" = apply(df, 2, median),
                          "SD of AbsErrors"  = apply(df, 2, sd),
                          "IQR of AbsErrors" = apply(df, 2, IQR),
                          "Min of AbsErrors" = apply(df, 2, min),
                          "Max of AbsErrors" = apply(df, 2, max))

rownames(models_comp) <- c("LM", "BestSub", "BestSubCV", "TrimmedBestSub")                        
View(models_comp)

#Boxplot of absolute errors
boxplot(df, main = "Abs. Errors Dist. of Models")

#Save the results--------------------------------
save(data2, train, trimmed_train, test, models_comp, file = "college_dataset_v1.R")


#Model 5: Ridge Regression------------------------
##Building Prediction Model----------------------
#Regularization
#Ridge Regression:
#The goal is to optimize:
#   RSS + lambda * Sum( beta_i ^ 2)
#   lambda => 0,  a tuning parameter

x <- model.matrix(Log_Apps ~ + . - Apps, data = train)[, -1] #remove intercept
y <- train$Log_Apps

lambda_ridge_grid <- 10 ^ seq(10, -2, length = 100)
lambda_ridge_grid

#Apply Ridge Regression
ridgereg_1 <- glmnet(x, y, alpha = 0, lambda = lambda_ridge_grid)
dim(coef(ridgereg_1))

#Plot Reg. Coefficients vs. Log Lambda
plot(ridgereg_1, xvar = "lambda")

#Retrieve Coefficients
ridgereg_1$lambda [50]
coef(ridgereg_1)[, 50]

#Cross validation to choose the best model
set.seed(1234)
ridge_cv    <- cv.glmnet(x, y, alpha = 0, nfolds = 10)
#The mean cross-validated error
ridge_cv$cvm
#Estimate of standard error of cvm.
ridge_cv$cvsd

#value of lambda that gives minimum cvm
ridge_cv$lambda.min

#Coefficients of regression w/ best_lambda
ridgereg_2 <- glmnet(x, y, alpha = 0, lambda = ridge_cv$lambda.min)
coef(ridgereg_2)

#Test the Model----------------------------------
#Prediction
#Create model matrix for test
test$Log_Apps <- log(test$Apps)
x_test <- model.matrix(Log_Apps ~ + . - Apps, data = test)[, -1]#remove intercept
pred_ridgereg <- predict(ridgereg_2, s = ridge_cv$lambda.min, newx = x_test)
pred_ridgereg
pred_ridgereg <- exp(pred_ridgereg)
pred_ridgereg
#Absolute error mean, median, sd, max, min-------
abs_err_ridgereg <- abs(pred_ridgereg - test$Apps)
models_comp <- rbind(models_comp, "RidgeReg" = c(mean(abs_err_ridgereg), 
                                                 median(abs_err_ridgereg),
                                                 sd(abs_err_ridgereg),
                                                 IQR(abs_err_ridgereg),
                                                 range(abs_err_ridgereg)))
View(models_comp)

#Actual vs. Predicted
plot(test$Apps, pred_ridgereg, xlab = "Actual", ylab = "Prediction")
abline(a = 0, b = 1, col = "red", lwd = 2)

#Why Does Ridge Regression Improve Over Least Squares?
#Model 6: Lasso Regression------------------------
#Regularization
#Lasso Regression:
#The goal is to optimize:
#   RSS + lambda * Sum(abs(beta_i))
#   lambda => 0,  a tuning parameter

#Apply Lasso Regression
lassoreg_1 <- glmnet(x, y, alpha = 1, lambda = lambda_ridge_grid)
dim(coef(lassoreg_1))

#Plot Reg. Coefficients vs. Log Lambda
plot(lassoreg_1, xvar = "lambda")

#Retrieve Coefficients
lassoreg_1$lambda [90]
coef(lassoreg_1)[, 90]

#Cross validation to choose the best model
set.seed(1234)
lasso_cv    <- cv.glmnet(x, y, alpha = 1, nfolds = 10)
#The mean cross-validated error
lasso_cv$cvm
#Estimate of standard error of cvm.
lasso_cv$cvsd

#value of lambda that gives minimum cvm
lasso_cv$lambda.min

#Coefficients of regression w/ best_lambda
lassoreg_2 <- glmnet(x, y, alpha = 1, lambda = lasso_cv$lambda.min)
coef(lassoreg_2)

#Test the Model----------------------------------
#Prediction
pred_lassoreg <- predict(lassoreg_2, s = lasso_cv$lambda.min, newx = x_test)
pred_lassoreg
pred_lassoreg <- exp(pred_lassoreg)
pred_lassoreg

#Absolute error mean, median, sd, max, min-------
abs_err_lassoreg <- abs(pred_lassoreg - test$Apps)
models_comp <- rbind(models_comp, "LassoReg" = c(mean(abs_err_lassoreg),
                                                 median(abs_err_lassoreg),
                                                 sd(abs_err_lassoreg),
                                                 IQR(abs_err_lassoreg),
                                                 range(abs_err_lassoreg)))
View(models_comp)

#Actual vs. Predicted
plot(test$Apps, pred_lassoreg, xlab = "Actual", ylab = "Prediction")
abline(a = 0, b = 1, col = "red", lwd = 2)

#Model 7: Decision Trees-------------------------
tree_1 <- rpart(Log_Apps ~ Accept + Top10perc + Grad.Rate , data = train, cp = 0.1, maxdepth = 3)

#Plot the tree
prp(tree_1)

#Decision Tree Structure
tree_1

#Change Complexity of Tree Model
tree_2 <- rpart(Log_Apps ~ Accept + Top10perc + Grad.Rate , data = train, cp = 0.001, maxdepth = 10)

#Plot the tree
prp(tree_2)
plotcp(tree_2)
tree_2$cptable
tree_2$cptable[which.min(tree_2$cptable[,"xerror"])]

#Prune the tree
tree_3 <- prune.rpart(tree_2, cp = tree_2$cptable[which.min(tree_2$cptable[,"xerror"])])

#Plot the pruned tree
prp(tree_3)

#Decision Tree Model Using All Variables
tree_4 <- rpart(formula = Log_Apps ~ . - Apps, data = train, cp = 0.0001, maxdepth = 20)

#Plot the tree
prp(tree_4)

#Prune the tree
plotcp(tree_4)
tree_4$cptable[which.min(tree_4$cptable[,"xerror"])]

#Prune the tree
tree_5 <- prune.rpart(tree_4, cp = tree_4$cptable[which.min(tree_4$cptable[,"xerror"])])

#Plot the tree
prp(tree_5)

#Test the Model----------------------------------
#Prediction:
pred_tree  <- predict(tree_5, test)
pred_tree  <- exp(pred_tree)
pred_tree

#Absolute error mean, median, sd, max, min-------
abs_err_tree <- abs(pred_tree - test$Apps)
models_comp <- rbind(models_comp, "Tree" = c(mean(abs_err_tree),
                                             median(abs_err_tree),
                                             sd(abs_err_tree),
                                             IQR(abs_err_tree),
                                             range(abs_err_tree)))
View(models_comp)

#Actual vs. Predicted
plot(test$Apps, pred_tree, xlab = "Actual", ylab = "Prediction")
abline(a = 0, b = 1, col = "red", lwd = 2)

#Model 8: Bagging--------------------------------
set.seed(1234)
bagging_1 <- randomForest(Log_Apps ~ . - Apps, mtry = ncol(train) - 2, ntree = 500, data = train)
bagging_1

#Test the Model----------------------------------
#Prediction: M8 Bagging
pred_bagging  <- predict(bagging_1, test)
pred_bagging  <- exp(pred_bagging)
pred_bagging

#Absolute error mean, median, sd, max, min-------
abs_err_bagging <- abs(pred_bagging - test$Apps)
models_comp <- rbind(models_comp, "Bagging" = c(mean(abs_err_bagging),
                                                median(abs_err_bagging),
                                                sd(abs_err_bagging),
                                                IQR(abs_err_bagging),
                                                range(abs_err_bagging)))
View(models_comp)

#Actual vs. Predicted
plot(test$Apps, pred_bagging, xlab = "Actual", ylab = "Prediction")
abline(a = 0, b = 1, col = "red", lwd = 2)

#Model 9: Random Forrest-------------------------
set.seed(1234)
rf_1 <- randomForest(Log_Apps ~ . - Apps, data = train, ntree = 500, importance = TRUE)
#mtry	
#for regression = p/3
rf_1

importance(rf_1)
varImpPlot(rf_1)

#%IncMSE: is based upon the mean decrease of accuracy in predictions on the out of bag samples when a given variable is excluded from the model. 
#IncNodePurity: is a measure of the total decrease in node impurity that results from splits over that variable, averaged over all trees. 
#The node impurity is measured by the training RSS.

#K-fold Cross-Validation for feature selection
#Don't forget to remove "Apps"[1] & "Log_Apps"[18]
#step = 0.75, 
#step 1: 16, step 2: example: round(0.75 * 16) = 12
#mtry: a function of number of remaining predictor variables to use 
#as the mtry parameter in the randomForest call
#example: default: floor(sqrt(p)), floor(p/3)
#recursive: whether variable importance is (re-)assessed at each step of variable reduction
set.seed(12345)
rf_cv <- rfcv(train[, - c(1, 18)], 
              train$Log_Apps, 
              cv.fold = 10,
              step = 0.75,
              mtry = function(p) max(1, floor(sqrt(p))),
              recursive = FALSE)
class(rf_cv)
str(rf_cv)
#Vector of number of variables used at each step
rf_cv$n.var
#Corresponding vector of MSEs at each step
rf_cv$error.cv
which.min(rf_cv$error.cv)

#Remove 7 variables based on Importance of Variables
sort(importance(rf_1)[,1])

#Regression formula
reg_formula <- as.formula(Log_Apps ~ Accept + Enroll + F.Undergrad + Outstate + Top10perc + Expend + Room.Board + Grad.Rate + Top25perc)
reg_formula
#mtry	
floor(sqrt(9))

set.seed(1234)
rf_2 <- randomForest(reg_formula, data = train, mtry = 3, ntree = 500)
rf_2

#Test the Model----------------------------------
#Prediction: M9 Random Forrest
pred_rf  <- predict(rf_2, test)
pred_rf  <- exp(pred_rf)
pred_rf

#Absolute error mean, median, sd, max, min-------
abs_err_rf <- abs(pred_rf - test$Apps)
models_comp <- rbind(models_comp, "RandomForrest" = c(mean(abs_err_rf),
                                                      median(abs_err_rf),
                                                      sd(abs_err_rf),
                                                      IQR(abs_err_rf),
                                                      range(abs_err_rf)))
View(models_comp)

#Actual vs. Predicted
plot(test$Salary, pred_rf, xlab = "Actual", ylab = "Prediction")
abline(a = 0, b = 1, col = "red", lwd = 2)

#Model 10: Bagging w/ Trimmed Train--------------
set.seed(1234)
trimmedbagging_1 <- randomForest(Log_Apps ~ . - Apps, mtry = ncol(trimmed_train) - 2, ntree = 500, data = trimmed_train)
trimmedbagging_1

#Test the Model----------------------------------
#Prediction:
pred_trimmedbagging  <- predict(trimmedbagging_1, test)
pred_trimmedbagging  <- exp(pred_trimmedbagging)
pred_trimmedbagging

#Absolute error mean, median, sd, max, min-------
abs_err_trimmedbagging <- abs(pred_trimmedbagging - test$Apps)
models_comp <- rbind(models_comp, "TrimmedBagging" = c(mean(abs_err_trimmedbagging),
                                                       median(abs_err_trimmedbagging),
                                                       sd(abs_err_trimmedbagging),
                                                       IQR(abs_err_trimmedbagging),
                                                       range(abs_err_trimmedbagging)))
View(models_comp)

#Actual vs. Predicted
plot(test$Apps, pred_trimmedbagging, xlab = "Actual", ylab = "Prediction")
abline(a = 0, b = 1, col = "red", lwd = 2)

#Save the results--------------------------------
save(data2, train, trimmed_train, test, models_comp, file = "case4_dataset_v2.R")

###End of Code###--------------------------------