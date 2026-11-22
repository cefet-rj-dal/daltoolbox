source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

data(Diabetes)
head(Diabetes)

set_example_seed()
sr <- train_test(sample_random(), Diabetes)
diabetes_train <- sr$train
diabetes_test <- sr$test

tune <- reg_tune(
  reg_svm("target"),
  ranges = list(
    epsilon = seq(0, 1, 0.2),
    cost = seq(20, 100, 20),
    kernel = c("radial")
  )
)

set_example_seed()
model <- fit(tune, diabetes_train)

train_prediction <- predict(model, diabetes_train)
train_eval <- evaluate(model, diabetes_train[, "target"], train_prediction)
train_eval$metrics

test_prediction <- predict(model, diabetes_test)
test_eval <- evaluate(model, diabetes_test[, "target"], test_prediction)
test_eval$metrics
