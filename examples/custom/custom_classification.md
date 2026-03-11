## Custom Classification

The primary goal of this example is to show how a classifier can be customized with very little integration code. The procedure is always the same: define a constructor to hold the configuration, implement `fit()` to train the model, implement `predict()` to return outputs in the expected format, and then reuse the same Experiment Line workflow for sampling and evaluation.

This is one of the main advantages of the framework: the complexity of the mining method stays in the external backend, while the complexity of integrating that method into the workflow stays low and predictable. In this concrete example, the custom classifier uses `RSNNS::mlp`.


``` r
# installation
# install.packages(c("daltoolbox", "RSNNS"))

library(daltoolbox)
```


``` r
cla_rsnns_custom <- function(attribute, slevels, size = 5, learn_rate = 0.2, maxit = 100) {
  obj <- daltoolbox::classification(attribute, slevels)
  obj$size <- size
  obj$learn_rate <- learn_rate
  obj$maxit <- maxit
  class(obj) <- append("cla_rsnns_custom", class(obj))
  obj
}

fit.cla_rsnns_custom <- function(obj, data, ...) {
  if (!requireNamespace("RSNNS", quietly = TRUE)) {
    stop("This example requires the 'RSNNS' package.")
  }

  data <- daltoolbox::adjust_data.frame(data)
  data[, obj$attribute] <- daltoolbox::adjust_factor(data[, obj$attribute], obj$ilevels, obj$slevels)
  obj$x <- setdiff(colnames(data), obj$attribute)

  x <- as.matrix(data[, obj$x, drop = FALSE])
  y <- daltoolbox::adjust_class_label(data[, obj$attribute])

  obj$model <- RSNNS::mlp(
    x = x,
    y = y,
    size = obj$size,
    learnFuncParams = c(obj$learn_rate),
    maxit = obj$maxit
  )

  obj
}

predict.cla_rsnns_custom <- function(object, x, ...) {
  x <- daltoolbox::adjust_data.frame(x)
  x <- as.matrix(x[, object$x, drop = FALSE])
  prediction <- predict(object$model, x)
  prediction <- as.data.frame(prediction)
  colnames(prediction) <- object$slevels
  prediction
}
```


``` r
iris <- datasets::iris
slevels <- levels(iris$Species)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test
```


``` r
model <- cla_rsnns_custom("Species", slevels, size = 5, learn_rate = 0.1, maxit = 150)
model <- fit(model, iris_train)

train_prediction <- predict(model, iris_train)
train_eval <- evaluate(model, iris_train$Species, train_prediction)
train_eval$metrics
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9583333 39 81  0  0         1      1           1           1  1
```


``` r
test_prediction <- predict(model, iris_test)
test_eval <- evaluate(model, iris_test$Species, test_prediction)
test_eval$metrics
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9333333 11 19  0  0         1      1           1           1  1
```

References
- Bergmeir, C., and Benitez, J. M. (2012). Neural Networks in R Using the Stuttgart Neural Network Simulator: RSNNS.
- Bishop, C. M. (1995). Neural Networks for Pattern Recognition.
