About the method
- `cla_bagging`: bootstrap aggregation of classification trees. The main idea is to reduce variance by fitting many trees on bootstrap samples and combining their votes.

Didactic goal: keep the same classification line of experiment used in the earlier examples and change only the learner family. This makes it easier to compare what bagging changes relative to a single-tree baseline.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "ipred"))

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
if (requireNamespace("ipred", quietly = TRUE)) {
  model <- cla_bagging("Species", nbagg = 25)
  set_example_seed()
  model <- fit(model, iris_train)
}
```

Training evaluation.

``` r
if (requireNamespace("ipred", quietly = TRUE)) {
  train_prediction <- predict(model, iris_train)
  train_eval <- evaluate(model, iris_train[, "Species"], train_prediction)
  train_eval$metrics
}
```

```
## NULL
```

Test evaluation.

``` r
if (requireNamespace("ipred", quietly = TRUE)) {
  test_prediction <- predict(model, iris_test)
  test_eval <- evaluate(model, iris_test[, "Species"], test_prediction)
  test_eval$metrics
}
```

```
## NULL
```

What to observe
- Bagging keeps the same DAL workflow while changing the model-construction logic internally.
- The learner usually becomes more stable than a single tree because the prediction is aggregated across bootstrap models.

References
- Breiman, L. (1996). Bagging Predictors.
