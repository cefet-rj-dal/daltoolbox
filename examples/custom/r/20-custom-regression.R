source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation
# install.packages(c("daltoolbox", "RSNNS", "MASS"))

library(daltoolbox)

reg_rsnns_custom <- function(attribute, size = 5, learn_rate = 0.1, maxit = 200) {
  obj <- daltoolbox::regression(attribute)
  obj$size <- size
  obj$learn_rate <- learn_rate
  obj$maxit <- maxit
  class(obj) <- append("reg_rsnns_custom", class(obj))
  obj
}

fit.reg_rsnns_custom <- function(obj, data, ...) {
  if (!requireNamespace("RSNNS", quietly = TRUE)) {
    stop("This example requires the 'RSNNS' package.")
  }

  data <- daltoolbox::adjust_data.frame(data)
  obj$x <- setdiff(colnames(data), obj$attribute)

  x <- as.matrix(data[, obj$x, drop = FALSE])
  y <- as.matrix(data[, obj$attribute, drop = FALSE])

  obj$model <- RSNNS::mlp(
    x = x,
    y = y,
    size = obj$size,
    learnFuncParams = c(obj$learn_rate),
    maxit = obj$maxit,
    linOut = TRUE
  )

  obj
}

predict.reg_rsnns_custom <- function(object, x, ...) {
  x <- daltoolbox::adjust_data.frame(x)
  x <- as.matrix(x[, object$x, drop = FALSE])
  as.numeric(predict(object$model, x))
}

library(MASS)
data(Diabetes)

set_example_seed()
sr <- sample_random()
sr <- train_test(sr, Diabetes)
diabetes_train <- sr$train
diabetes_test <- sr$test

model <- reg_rsnns_custom("target", size = 6, learn_rate = 0.05, maxit = 200)
set_example_seed()
model <- fit(model, diabetes_train)

train_prediction <- predict(model, diabetes_train)
train_eval <- evaluate(model, diabetes_train[, "target"], train_prediction)
train_eval$metrics

test_prediction <- predict(model, diabetes_test)
test_eval <- evaluate(model, diabetes_test[, "target"], test_prediction)
test_eval$metrics
