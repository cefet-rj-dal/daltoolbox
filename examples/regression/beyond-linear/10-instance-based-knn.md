About the method
- `reg_knn`: k-Nearest Neighbors for regression. Predicts a value from the target values of nearby cases.
- Main parameter: `k`.

Didactic goal: keep the same regression line of experiment and change only the learner to a local neighborhood method.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```

Load data and inspect.

``` r
data(Diabetes)
head(Diabetes)
```

```
##            age         sex         bmi           bp           s1          s2
## 1  0.038075906  0.05068012  0.06169621  0.021872386 -0.044223498 -0.03482076
## 2 -0.001882017 -0.04464164 -0.05147406 -0.026327528 -0.008448724 -0.01916334
## 3  0.085298906  0.05068012  0.04445121 -0.005670422 -0.045599451 -0.03419447
## 4 -0.089062939 -0.04464164 -0.01159501 -0.036656081  0.012190569  0.02499059
## 5  0.005383060 -0.04464164 -0.03638469  0.021872386  0.003934852  0.01559614
## 6 -0.092695478 -0.04464164 -0.04069594 -0.019441826 -0.068990650 -0.07928784
##             s3           s4           s5           s6 target
## 1 -0.043400846 -0.002592262  0.019907486 -0.017646125    151
## 2  0.074411564 -0.039493383 -0.068331547 -0.092204050     75
## 3 -0.032355932 -0.002592262  0.002861309 -0.025930339    141
## 4 -0.036037570  0.034308859  0.022687745 -0.009361911    206
## 5  0.008142084 -0.002592262 -0.031987639 -0.046640874    135
## 6  0.041276824 -0.076394504 -0.041176167 -0.096346157     97
```

Target `target` and reproducible train/test split.

``` r
set_example_seed()
sr <- train_test(sample_random(), Diabetes)
diabetes_train <- sr$train
diabetes_test <- sr$test
```

Model configuration and fitting.

``` r
model <- reg_knn("target", k = 5)
set_example_seed()
model <- fit(model, diabetes_train)
```

Training evaluation.

``` r
train_prediction <- predict(model, diabetes_train)
train_eval <- evaluate(model, diabetes_train[, "target"], train_prediction)
train_eval$metrics
```

```
##        mse     smape       R2
## 1 2377.263 0.2778633 0.617395
```

Test evaluation.

``` r
test_prediction <- predict(model, diabetes_test)
test_eval <- evaluate(model, diabetes_test[, "target"], test_prediction)
test_eval$metrics
```

```
##        mse     smape        R2
## 1 3420.975 0.3496625 0.2518549
```

References
- Altman, N. (1992). An Introduction to Kernel and Nearest-Neighbor Nonparametric Regression.
