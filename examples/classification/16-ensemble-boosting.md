About the method
- `cla_boosting`: boosting classifier that builds a sequence of weak learners, each one focusing more on previously difficult cases.

Didactic goal: keep the line of experiment fixed and isolate the change in modeling idea. This helps the reader compare boosting with bagging and random forests without changing the data or evaluation structure.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "adabag"))

library(daltoolbox)
library(adabag)
```

```
## Loading required package: rpart
```

```
## Loading required package: caret
```

```
## Loading required package: ggplot2
```

```
## Loading required package: lattice
```

```
## 
## Attaching package: 'caret'
```

```
## The following object is masked from 'package:daltoolbox':
## 
##     cluster
```

```
## Loading required package: foreach
```

```
## Loading required package: doParallel
```

```
## Loading required package: iterators
```

```
## Loading required package: parallel
```

```
## 
## Attaching package: 'adabag'
```

```
## The following object is masked from 'package:ipred':
## 
##     bagging
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
model <- cla_boosting("Species", mfinal = 20)
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
##   accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1        1 41 79  0  0         1      1           1           1  1
```

Test evaluation.

``` r
test_prediction <- predict(model, iris_test)
test_eval <- evaluate(model, iris_test[, "Species"], test_prediction)
test_eval$metrics
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9333333  9 21  0  0         1      1           1           1  1
```

What to observe
- Boosting changes the learner sequentially, not only by averaging independent resamples.
- The experiment body remains the same even though the ensemble logic differs from bagging and forests.

References
- Freund, Y., and Schapire, R. E. (1997). A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting.
