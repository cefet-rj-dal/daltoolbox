# Regression tuning 

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

# Dataset for regression analysis

library(MASS)
data(Boston)
print(t(sapply(Boston, class)))
head(Boston)

# for performance, you can convert to matrix
Boston <- as.matrix(Boston)

# preparing dataset for random sampling
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, Boston)
boston_train <- sr$train
boston_test <- sr$test

# Training

tune <- reg_tune(reg_svm("medv"), 
          ranges = list(seq(0,1,0.2), cost=seq(20,100,20), kernel = c("radial")))
model <- fit(tune, boston_train)

# Model adjustment

train_prediction <- predict(model, boston_train)
boston_train_predictand <- boston_train[,"medv"]
train_eval <- evaluate(model, boston_train_predictand, train_prediction)
print(train_eval$metrics)

# Test

test_prediction <- predict(model, boston_test)
boston_test_predictand <- boston_test[,"medv"]
test_eval <- evaluate(model, boston_test_predictand, test_prediction)
print(test_eval$metrics)

# Options for other models

# svm
ranges <- list(seq(0,1,0.2), cost=seq(20,100,20), kernel = c("linear", "radial", "polynomial", "sigmoid"))

# knn
ranges <- list(k=1:20)

# mlp
ranges <- list(size=1:10, decay=seq(0, 1, 0.1))

# rf
ranges <- list(mtry=1:10, ntree=1:10)
