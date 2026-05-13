About the method
- `cla_dtree`: decision tree for classification. Recursively splits explanatory variables to separate classes, yielding an interpretable model.

Didactic goal: keep exactly the same classification line of experiment introduced by the baseline and change only the learner to an interpretable supervised model.

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
model <- cla_dtree("Species", slevels)
set_example_seed()
model <- fit(model, iris_train)
```

Training evaluation.

``` r
train_prediction <- predict(model, iris_train)
train_eval <- evaluate(model, iris_train[, "Species"], train_prediction)
train_eval$metrics
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9666667 41 79  0  0         1      1           1           1  1
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
- The experiment structure is unchanged from the baseline example.
- The difference now is that the learner can express feature-based decision rules instead of only the majority class.

References
- Breiman, L., Friedman, J., Olshen, R., and Stone, C. (1984). Classification and Regression Trees.
