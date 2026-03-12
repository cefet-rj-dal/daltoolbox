## Tutorial 06 - Metrics and Evaluation

Running `evaluate()` is easy; interpreting the result is where learning starts. This tutorial slows down the evaluation step and explains why training and test metrics should be read together.

The goal is to help the reader connect the numbers to a model's behavior instead of treating metrics as output to copy and paste.


``` r
# install.packages("daltoolbox")

library(daltoolbox)
```

Train a decision tree on a standard split so we can focus on the evaluation stage.

``` r
iris <- datasets::iris
slevels <- levels(iris$Species)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)

model <- cla_dtree("Species", slevels)
model <- fit(model, sr$train)
```

Evaluate on the training set first. These numbers usually look optimistic because the model has already seen these cases.

``` r
train_prediction <- predict(model, sr$train)
train_eval <- evaluate(model, adjust_class_label(sr$train$Species), train_prediction)
train_eval$metrics
```

```
##   accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1    0.975 39 81  0  0         1      1           1           1  1
```

Now evaluate on the test set. This is the more important number for judging whether the model is likely to generalize.

``` r
test_prediction <- predict(model, sr$test)
test_eval <- evaluate(model, adjust_class_label(sr$test$Species), test_prediction)
test_eval$metrics
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9666667 11 19  0  0         1      1           1           1  1
```

It is also useful to inspect the confusion matrix. While a single score summarizes performance, the confusion matrix shows which classes are being confused with one another.

``` r
table(predicted = test_eval$prediction, observed = sr$test$Species)
```

```
##             observed
## predicted    setosa versicolor virginica
##   setosa         11          0         0
##   versicolor      0         13         0
##   virginica       0          1         5
```

For beginners, one interpretation rule helps a lot: if training results are much better than test results, the model may be too adapted to the training data.
