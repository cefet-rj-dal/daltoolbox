source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "xgboost"))

library(daltoolbox)

iris <- datasets::iris
head(iris)

slevels <- levels(iris$Species)

set_example_seed()
sr <- train_test(sample_random(), iris)
iris_train <- sr$train
iris_test <- sr$test

tbl <- rbind(
  table(iris[, "Species"]),
  table(iris_train[, "Species"]),
  table(iris_test[, "Species"])
)
rownames(tbl) <- c("dataset", "training", "test")
tbl

if (requireNamespace("xgboost", quietly = TRUE)) {
  model <- cla_xgboost(
    "Species",
    params = list(max_depth = 2, eta = 0.2, nthread = 1),
    nrounds = 5
  )
  model <- fit(model, iris_train)
}

if (requireNamespace("xgboost", quietly = TRUE)) {
  train_prediction <- predict(model, iris_train)
  train_eval <- evaluate(model, iris_train[, "Species"], train_prediction)
  train_eval$metrics
}

if (requireNamespace("xgboost", quietly = TRUE)) {
  test_prediction <- predict(model, iris_test)
  test_eval <- evaluate(model, iris_test[, "Species"], test_prediction)
  test_eval$metrics
}
