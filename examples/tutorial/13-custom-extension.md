## Tutorial 13 - Custom Extension

One of the strongest ideas in `daltoolbox` is extensibility. A researcher may want to use a method that is not built into the package, but still keep the same Experiment Line workflow.

This tutorial revisits that idea with a compact custom classifier based on `RSNNS::mlp`. The important lesson is not the specific backend; it is the integration pattern.


``` r
# install.packages(c("daltoolbox", "RSNNS"))

library(daltoolbox)
```

Define a constructor that stores the learner configuration. This is the object the user will instantiate before training.

``` r
cla_rsnns_custom <- function(attribute, slevels, size = 5, learn_rate = 0.2, maxit = 100) {
  obj <- daltoolbox::classification(attribute, slevels)
  obj$size <- size
  obj$learn_rate <- learn_rate
  obj$maxit <- maxit
  class(obj) <- append("cla_rsnns_custom", class(obj))
  obj
}
```

Implement `fit()` so the custom object can be trained inside the same workflow used by built-in learners.

``` r
fit.cla_rsnns_custom <- function(obj, data, ...) {
  if (!requireNamespace("RSNNS", quietly = TRUE)) {
    stop("This tutorial requires the 'RSNNS' package.")
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
```

Implement `predict()` so the fitted object returns outputs in the format expected by the framework.

``` r
predict.cla_rsnns_custom <- function(object, x, ...) {
  x <- daltoolbox::adjust_data.frame(x)
  x <- as.matrix(x[, object$x, drop = FALSE])
  prediction <- predict(object$model, x)
  prediction <- as.data.frame(prediction)
  colnames(prediction) <- object$slevels
  prediction
}
```

Finally, use the custom learner exactly as you would use a built-in learner. This is the pedagogical payoff of the design.

``` r
iris <- datasets::iris
slevels <- levels(iris$Species)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)

model <- cla_rsnns_custom("Species", slevels, size = 5, learn_rate = 0.1, maxit = 150)
model <- fit(model, sr$train)

prediction <- predict(model, sr$test)
eval <- evaluate(model, sr$test$Species, prediction)
eval$metrics
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9666667 11 19  0  0         1      1           1           1  1
```

The core message is that extending the framework should not require rewriting the entire experimental workflow.
