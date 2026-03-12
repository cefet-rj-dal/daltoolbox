# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
head(iris)

slevels <- levels(iris$Species)
slevels

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)

iris_train <- sr$train
iris_test <- sr$test

model <- cla_dtree("Species", slevels)
model <- fit(model, iris_train)

train_prediction <- predict(model, iris_train)
train_eval <- evaluate(model, adjust_class_label(iris_train$Species), train_prediction)
train_eval$metrics

test_prediction <- predict(model, iris_test)
test_eval <- evaluate(model, adjust_class_label(iris_test$Species), test_prediction)
test_eval$metrics
