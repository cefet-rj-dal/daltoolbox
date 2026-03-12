## Tutorial 10 - Regression Workflow

The Experiment Line idea is not limited to classification. In regression, the target becomes numeric, but the main stages remain the same: split, fit, predict, and evaluate.

This tutorial helps the learner see that the framework is stable even when the analytical task changes.


``` r
# install.packages("daltoolbox")

library(daltoolbox)
library(MASS)
```

Load a numeric prediction problem. The goal is to predict `medv`, a continuous outcome.

``` r
data(Boston)
head(Boston)
```

```
##      crim zn indus chas   nox    rm  age    dis rad tax ptratio  black lstat
## 1 0.00632 18  2.31    0 0.538 6.575 65.2 4.0900   1 296    15.3 396.90  4.98
## 2 0.02731  0  7.07    0 0.469 6.421 78.9 4.9671   2 242    17.8 396.90  9.14
## 3 0.02729  0  7.07    0 0.469 7.185 61.1 4.9671   2 242    17.8 392.83  4.03
## 4 0.03237  0  2.18    0 0.458 6.998 45.8 6.0622   3 222    18.7 394.63  2.94
## 5 0.06905  0  2.18    0 0.458 7.147 54.2 6.0622   3 222    18.7 396.90  5.33
## 6 0.02985  0  2.18    0 0.458 6.430 58.7 6.0622   3 222    18.7 394.12  5.21
##   medv
## 1 24.0
## 2 21.6
## 3 34.7
## 4 33.4
## 5 36.2
## 6 28.7
```

``` r
Boston <- as.matrix(Boston)
```

Create train and test partitions.

``` r
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, Boston)

boston_train <- sr$train
boston_test <- sr$test
```

Fit a regression tree and inspect both training and test errors.

``` r
model <- reg_dtree("medv")
model <- fit(model, boston_train)

train_prediction <- predict(model, boston_train)
train_eval <- evaluate(model, boston_train[, "medv"], train_prediction)
train_eval$metrics
```

```
##        mse     smape        R2
## 1 12.68065 0.1345098 0.8591168
```

``` r
test_prediction <- predict(model, boston_test)
test_eval <- evaluate(model, boston_test[, "medv"], test_prediction)
test_eval$metrics
```

```
##        mse     smape        R2
## 1 29.38142 0.1642396 0.5117372
```

From a teaching perspective, this is a useful turning point: the workflow is familiar, but the interpretation of the metrics now belongs to regression rather than classification.
