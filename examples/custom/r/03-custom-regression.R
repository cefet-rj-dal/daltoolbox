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
data(Boston)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, Boston)
boston_train <- sr$train
boston_test <- sr$test

model <- reg_rsnns_custom("medv", size = 6, learn_rate = 0.05, maxit = 200)
model <- fit(model, boston_train)

train_prediction <- predict(model, boston_train)
train_eval <- evaluate(model, boston_train[, "medv"], train_prediction)
train_eval$metrics

test_prediction <- predict(model, boston_test)
test_eval <- evaluate(model, boston_test[, "medv"], test_prediction)
test_eval$metrics
