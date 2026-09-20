About the utility
- `cla_tune`: performs hyperparameter search for a classifier over ranges defined in `ranges`.
- In this example the tuned base learner is `cla_svm`.

Didactic goal: preserve the same classification line of experiment and treat tuning as a change in model configuration rather than as a separate workflow.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```

Load data and inspect.

``` r
iris <- datasets::iris
head(iris)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
## 1          5.1         3.5          1.4         0.2  setosa
## 2          4.9         3.0          1.4         0.2  setosa
## 3          4.7         3.2          1.3         0.2  setosa
## 4          4.6         3.1          1.5         0.2  setosa
## 5          5.0         3.6          1.4         0.2  setosa
## 6          5.4         3.9          1.7         0.4  setosa
```

Target `Species` levels and reproducible train/test split.

``` r
slevels <- levels(iris$Species)

set_example_seed()
sr <- train_test(sample_random(), iris)
iris_train <- sr$train
iris_test <- sr$test
```

Class distribution by split.

``` r
tbl <- rbind(
  table(iris[, "Species"]),
  table(iris_train[, "Species"]),
  table(iris_test[, "Species"])
)
rownames(tbl) <- c("dataset", "training", "test")
tbl
```

```
##          setosa versicolor virginica
## dataset      50         50        50
## training     41         39        40
## test          9         11        10
```

Model configuration and fitting.

``` r
tune <- cla_tune(
  cla_svm("Species", slevels),
  ranges = list(
    epsilon = seq(0, 1, 0.2),
    cost = seq(20, 100, 20),
    kernel = c("linear", "radial", "polynomial", "sigmoid")
  )
)

set_example_seed()
model <- fit(tune, iris_train)
```

Training evaluation.

``` r
train_prediction <- predict(model, iris_train)
train_eval <- evaluate(model, iris_train[, "Species"], train_prediction)
train_eval$metrics
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9833333 41 79  0  0         1      1           1           1  1
```

Test evaluation.

``` r
test_prediction <- predict(model, iris_test)
test_eval <- evaluate(model, iris_test[, "Species"], test_prediction)
test_eval$metrics
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9666667  9 21  0  0         1      1           1           1  1
```

What to observe
- The experiment body is still split, fit, predict, and evaluate.
- Tuning changes how the model is selected, not how the experiment is read.

References
- Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.
