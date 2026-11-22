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
tune <- reg_tune(
  reg_svm("target"),
  ranges = list(
    epsilon = seq(0, 1, 0.2),
    cost = seq(20, 100, 20),
    kernel = c("radial")
  )
)

set_example_seed()
model <- fit(tune, diabetes_train)
```

Training evaluation.

``` r
train_prediction <- predict(model, diabetes_train)
train_eval <- evaluate(model, diabetes_train[, "target"], train_prediction)
train_eval$metrics
```

```
##        mse     smape        R2
## 1 2864.137 0.3450165 0.5390358
```

Test evaluation.

``` r
test_prediction <- predict(model, diabetes_test)
test_eval <- evaluate(model, diabetes_test[, "target"], test_prediction)
test_eval$metrics
```

```
##        mse     smape        R2
## 1 3804.902 0.3880649 0.1678926
```

What to observe
- The workflow is still split, fit, predict, and evaluate.
- Tuning changes how the regressor is chosen, not how the line of experiment is structured.

References
- Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.
