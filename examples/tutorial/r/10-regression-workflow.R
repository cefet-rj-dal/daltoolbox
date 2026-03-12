# install.packages("daltoolbox")

library(daltoolbox)
library(MASS)

data(Boston)
head(Boston)

Boston <- as.matrix(Boston)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, Boston)

boston_train <- sr$train
boston_test <- sr$test

model <- reg_dtree("medv")
model <- fit(model, boston_train)

train_prediction <- predict(model, boston_train)
train_eval <- evaluate(model, boston_train[, "medv"], train_prediction)
train_eval$metrics

test_prediction <- predict(model, boston_test)
test_eval <- evaluate(model, boston_test[, "medv"], test_prediction)
test_eval$metrics
