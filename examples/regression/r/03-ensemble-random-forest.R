# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

library(MASS)
data(Boston)
print(t(sapply(Boston, class)))
head(Boston)

# for performance, you can convert to matrix
Boston <- as.matrix(Boston)

# preparing random sampling
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, Boston)
boston_train <- sr$train
boston_test <- sr$test

model <- reg_rf("medv", mtry=7, ntree=30) # mtry: variables per split; ntree: number of trees
model <- fit(model, boston_train)

train_prediction <- predict(model, boston_train)
boston_train_predictand <- boston_train[,"medv"]
train_eval <- evaluate(model, boston_train_predictand, train_prediction)
print(train_eval$metrics)

test_prediction <- predict(model, boston_test)
boston_test_predictand <- boston_test[,"medv"]
test_eval <- evaluate(model, boston_test_predictand, test_prediction)
print(test_eval$metrics)
