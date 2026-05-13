About the utility
- `reg_tune`: hyperparameter search for regression models over ranges defined in `ranges`.
- In this example the tuned base learner is `reg_svm`.

Didactic goal: preserve the same regression line of experiment and treat tuning as a change in model configuration rather than as a separate workflow.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```

Load data and inspect.

``` r
data(Boston)
head(Boston)
```

```
##      crim zn indus chas   nox    rm  age    dis rad tax ptratio  black lstat medv
## 1 0.00632 18  2.31    0 0.538 6.575 65.2 4.0900   1 296    15.3 396.90  4.98 24.0
## 2 0.02731  0  7.07    0 0.469 6.421 78.9 4.9671   2 242    17.8 396.90  9.14 21.6
## 3 0.02729  0  7.07    0 0.469 7.185 61.1 4.9671   2 242    17.8 392.83  4.03 34.7
## 4 0.03237  0  2.18    0 0.458 6.998 45.8 6.0622   3 222    18.7 394.63  2.94 33.4
## 5 0.06905  0  2.18    0 0.458 7.147 54.2 6.0622   3 222    18.7 396.90  5.33 36.2
## 6 0.02985  0  2.18    0 0.458 6.430 58.7 6.0622   3 222    18.7 394.12  5.21 28.7
```

Target `medv` and reproducible train/test split.

``` r
set_example_seed()
sr <- train_test(sample_random(), Boston)
boston_train <- sr$train
boston_test <- sr$test
```

Model configuration and fitting.

``` r
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
```

Training evaluation.

``` r
train_prediction <- predict(model, boston_train)
train_eval <- evaluate(model, boston_train[, "medv"], train_prediction)
train_eval$metrics
```

```
##        mse      smape        R2
## 1 3.093724 0.07190626 0.9611203
```

Test evaluation.

``` r
test_prediction <- predict(model, boston_test)
test_eval <- evaluate(model, boston_test[, "medv"], test_prediction)
test_eval$metrics
```

```
##        mse    smape        R2
## 1 9.927719 0.105222 0.9033131
```

What to observe
- The workflow is still split, fit, predict, and evaluate.
- Tuning changes how the regressor is chosen, not how the line of experiment is structured.

References
- Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.
