# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
slevels <- levels(iris$Species)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)

model <- cla_dtree("Species", slevels)
model <- fit(model, sr$train)

train_prediction <- predict(model, sr$train)
train_eval <- evaluate(model, adjust_class_label(sr$train$Species), train_prediction)
train_eval$metrics

test_prediction <- predict(model, sr$test)
test_eval <- evaluate(model, adjust_class_label(sr$test$Species), test_prediction)
test_eval$metrics

table(predicted = test_prediction, observed = sr$test$Species)
