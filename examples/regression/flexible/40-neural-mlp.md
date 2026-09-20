About the method
- `reg_mlp`: Multilayer Perceptron for regression.
- Main parameters: `size` and `decay`.

Didactic goal: keep the same regression line of experiment and change only the learner to a neural regressor.

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

Target `medv` and reproducible train/test split.

``` r
set_example_seed()
sr <- train_test(sample_random(), Boston)
boston_train <- sr$train
boston_test <- sr$test
```

Model configuration and fitting.

``` r
model <- reg_mlp("medv", size = 5, decay = 0.54)
set_example_seed()
model <- fit(model, boston_train)
```

Training evaluation.

``` r
train_prediction <- predict(model, boston_train)
train_eval <- evaluate(model, boston_train[, "medv"], train_prediction)
train_eval$metrics
```

```
##        mse     smape        R2
## 1 8.538853 0.1016522 0.8926898
```

Test evaluation.

``` r
test_prediction <- predict(model, boston_test)
test_eval <- evaluate(model, boston_test[, "medv"], test_prediction)
test_eval$metrics
```

```
##        mse     smape        R2
## 1 11.36168 0.1142616 0.8893476
```

References
- Bishop, C. M. (1995). Neural Networks for Pattern Recognition.
