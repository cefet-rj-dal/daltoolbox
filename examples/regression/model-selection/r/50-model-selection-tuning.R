source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

data(Boston)
head(Boston)

set_example_seed()
sr <- train_test(sample_random(), Boston)
boston_train <- sr$train
boston_test <- sr$test

tune <- reg_tune(
  reg_svm("medv"),
  ranges = list(
    epsilon = seq(0, 1, 0.2),
    cost = seq(20, 100, 20),
    kernel = c("radial")
  )
)

set_example_seed()
model <- fit(tune, boston_train)

train_prediction <- predict(model, boston_train)
train_eval <- evaluate(model, boston_train[, "medv"], train_prediction)
train_eval$metrics

test_prediction <- predict(model, boston_test)
test_eval <- evaluate(model, boston_test[, "medv"], test_prediction)
test_eval$metrics
