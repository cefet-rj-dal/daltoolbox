## Custom Regression

The primary goal of this example is to show how to build a custom regressor that remains compatible with the same Experiment Line used by the native learners of `daltoolbox`. The customization procedure is straightforward: define a constructor, keep the hyperparameters inside the object, implement `fit()` and `predict()`, and then reuse the standard sampling and `evaluate()` steps.

This makes the separation of concerns explicit. The regression algorithm can be complex, but the integration contract is intentionally small. In this concrete example, the custom regressor uses `RSNNS::mlp` with linear output.


``` r
# installation
# install.packages(c("daltoolbox", "RSNNS", "MASS"))

library(daltoolbox)
```


``` r
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
```


``` r
library(MASS)
data(Boston)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, Boston)
boston_train <- sr$train
boston_test <- sr$test
```


``` r
model <- reg_rsnns_custom("medv", size = 6, learn_rate = 0.05, maxit = 200)
model <- fit(model, boston_train)

train_prediction <- predict(model, boston_train)
train_eval <- evaluate(model, boston_train[, "medv"], train_prediction)
train_eval$metrics
```

```
##        mse     smape       R2
## 1 99.57521 0.3467951 -0.10629
```


``` r
test_prediction <- predict(model, boston_test)
test_eval <- evaluate(model, boston_test[, "medv"], test_prediction)
test_eval$metrics
```

```
##        mse    smape         R2
## 1 82.43641 0.351738 -0.3699349
```

References
- Bergmeir, C., and Benitez, J. M. (2012). Neural Networks in R Using the Stuttgart Neural Network Simulator: RSNNS.
- Bishop, C. M. (1995). Neural Networks for Pattern Recognition.
