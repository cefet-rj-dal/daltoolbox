# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
slevels <- levels(iris$Species)

set.seed(1)
sr <- train_test(sample_random(), iris)
iris_train <- sr$train
iris_test <- sr$test

tune <- cla_tune(
  cla_svm("Species", slevels),
  ranges = list(
    epsilon = seq(0, 0.4, 0.2),
    cost = seq(20, 60, 20),
    kernel = c("linear", "radial")
  )
)

model <- fit(tune, iris_train)

train_prediction <- predict(model, iris_train)
train_eval <- evaluate(model, adjust_class_label(iris_train$Species), train_prediction)
train_eval$metrics

test_prediction <- predict(model, iris_test)
test_eval <- evaluate(model, adjust_class_label(iris_test$Species), test_prediction)
test_eval$metrics
