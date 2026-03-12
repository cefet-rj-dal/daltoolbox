## Tutorial 01 - First Experiment

The first objective is simple: understand the basic rhythm of a `daltoolbox` experiment. In most supervised tasks, the same cycle appears again and again: prepare the data, split it, configure a learner, fit the model, predict on unseen cases, and evaluate the result.

This tutorial intentionally uses a small and familiar dataset so the reader can focus on the workflow rather than on the domain.


``` r
# install.packages("daltoolbox")

library(daltoolbox)
```

Start by loading a dataset and identifying the target variable. In classification, it is important to know the possible class labels before training a learner.

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

``` r
slevels <- levels(iris$Species)
slevels
```

```
## [1] "setosa"     "versicolor" "virginica"
```

Now create a training set and a test set. The training data is used to fit the learner; the test data is used later to estimate how the model behaves on new cases.

``` r
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)

iris_train <- sr$train
iris_test <- sr$test
```

With the split ready, define a first learner. A decision tree is a good starting point because it is fast and easy to interpret.

``` r
model <- cla_dtree("Species", slevels)
model <- fit(model, iris_train)
```

After training, always check performance on both the training data and the test data. The comparison helps reveal whether the learner is simply memorizing the training set or generalizing in a reasonable way.

``` r
train_prediction <- predict(model, iris_train)
train_eval <- evaluate(model, adjust_class_label(iris_train$Species), train_prediction)
train_eval$metrics
```

```
##   accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1    0.975 39 81  0  0         1      1           1           1  1
```

``` r
test_prediction <- predict(model, iris_test)
test_eval <- evaluate(model, adjust_class_label(iris_test$Species), test_prediction)
test_eval$metrics
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9666667 11 19  0  0         1      1           1           1  1
```

This structure is the foundation of the package. The later tutorials mainly change the sampling strategy, the preprocessing steps, or the learner itself, but the overall logic remains stable.
