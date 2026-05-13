About the method
- `cla_glm`: binary logistic regression using the generalized linear model framework.

Didactic goal: preserve the same line of experiment and change only what the method requires. In this case, the special requirement is a binary target instead of the original three-class `Species` problem.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```

Load data, derive a binary target, and inspect.

``` r
iris_bin <- datasets::iris
iris_bin$IsVersicolor <- factor(ifelse(
  iris_bin$Species == "versicolor",
  "versicolor",
  "not_versicolor"
))
head(iris_bin)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species   IsVersicolor
## 1          5.1         3.5          1.4         0.2  setosa not_versicolor
## 2          4.9         3.0          1.4         0.2  setosa not_versicolor
## 3          4.7         3.2          1.3         0.2  setosa not_versicolor
## 4          4.6         3.1          1.5         0.2  setosa not_versicolor
## 5          5.0         3.6          1.4         0.2  setosa not_versicolor
## 6          5.4         3.9          1.7         0.4  setosa not_versicolor
```

Target levels and reproducible stratified split.

``` r
slevels <- levels(iris_bin$IsVersicolor)

set_example_seed()
sr <- train_test(sample_stratified("IsVersicolor"), iris_bin)
iris_train <- sr$train
iris_test <- sr$test
```

Class distribution by split.

``` r
tbl <- rbind(
  table(iris_bin[, "IsVersicolor"]),
  table(iris_train[, "IsVersicolor"]),
  table(iris_test[, "IsVersicolor"])
)
rownames(tbl) <- c("dataset", "training", "test")
tbl
```

```
##          not_versicolor versicolor
## dataset             100         50
## training             80         40
## test                 20         10
```

Model configuration and fitting.

``` r
model <- cla_glm("IsVersicolor", positive = "versicolor")
model <- suppressWarnings(fit(model, iris_train))
```

Training evaluation.

``` r
train_prediction <- predict(model, iris_train)
train_eval <- evaluate(model, iris_train[, "IsVersicolor"], train_prediction)
train_eval$metrics
```

```
##   accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1        1 80 40  0  0         1      1           1           1  1
```

Test evaluation.

``` r
test_prediction <- predict(model, iris_test)
test_eval <- evaluate(model, iris_test[, "IsVersicolor"], test_prediction)
test_eval$metrics
```

```
##   accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1        1 20 10  0  0         1      1           1           1  1
```

What to observe
- The experiment body is the same as in multiclass classification.
- The only structural adaptation is the binary target required by logistic regression.

References
- McCullagh, P., and Nelder, J. A. (1989). Generalized Linear Models.
