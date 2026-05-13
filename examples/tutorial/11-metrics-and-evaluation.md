## Tutorial 11 - Metrics and Evaluation

Running `evaluate()` is easy; interpreting the result is where learning starts. This tutorial slows down the evaluation step and explains why training and test metrics should be read together.

The goal is to help the reader connect the numbers to a model's behavior instead of treating metrics as output to copy and paste.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```

Train a decision tree on a standard split so we can focus on the evaluation stage.

``` r
iris <- datasets::iris
slevels <- levels(iris$Species)

set_example_seed()
sr <- sample_random()
sr <- train_test(sr, iris)

model <- cla_dtree("Species", slevels)
set_example_seed()
model <- fit(model, sr$train)
```

Evaluate on the training set first. These numbers usually look optimistic because the model has already seen these cases. The prediction object is a class-score table, and `evaluate()` converts it into labels and metrics.

``` r
train_prediction <- predict(model, sr$train)
train_eval <- evaluate(model, sr$train$Species, train_prediction)
train_eval$metrics
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9666667 41 79  0  0         1      1           1           1  1
```

Now evaluate on the test set. This is the more important number for judging whether the model is likely to generalize.

``` r
test_prediction <- predict(model, sr$test)
test_eval <- evaluate(model, sr$test$Species, test_prediction)
test_eval$metrics
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9666667  9 21  0  0         1      1           1           1  1
```

It is also useful to inspect the confusion matrix. While a single score summarizes performance, the confusion matrix shows which classes are being confused with one another.

``` r
table(predicted = test_eval$prediction, observed = sr$test$Species)
```

```
##             observed
## predicted    setosa versicolor virginica
##   setosa          9          0         0
##   versicolor      0         10         0
##   virginica       0          1        10
```

For beginners, one interpretation rule helps a lot: if training results are much better than test results, the model may be too adapted to the training data.
