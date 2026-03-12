## Tutorial 08 - Tuning Workflow

After comparing default learners, it becomes natural to ask whether one of them can be improved by hyperparameter search. Tuning is useful, but it should come after the experimental basics are already under control.

This tutorial shows tuning as a disciplined extension of the same workflow, not as a separate world.


``` r
# install.packages("daltoolbox")

library(daltoolbox)
```

Prepare the classification problem.

``` r
iris <- datasets::iris
slevels <- levels(iris$Species)

set.seed(1)
sr <- train_test(sample_random(), iris)
iris_train <- sr$train
iris_test <- sr$test
```

Define a compact search space. The example is intentionally small so the reader can understand the structure of tuning without turning the tutorial into a long computation.

``` r
tune <- cla_tune(
  cla_svm("Species", slevels),
  ranges = list(
    epsilon = seq(0, 0.4, 0.2),
    cost = seq(20, 60, 20),
    kernel = c("linear", "radial")
  )
)

model <- fit(tune, iris_train)
```

Check the tuned model on both the training and test data. As before, the interpretation depends on comparing both views of performance.

``` r
train_prediction <- predict(model, iris_train)
train_eval <- evaluate(model, adjust_class_label(iris_train$Species), train_prediction)
train_eval$metrics
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9833333 39 81  0  0         1      1           1           1  1
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

A good learning takeaway is that tuning should refine a solid workflow, not replace careful sampling and evaluation.
