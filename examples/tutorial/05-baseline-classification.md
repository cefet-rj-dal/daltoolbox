## Tutorial 05 - Baseline Classification

A baseline is one of the healthiest habits in experimental work. Before testing advanced learners, it is useful to establish a simple reference that answers a basic question: is the stronger model actually adding value?

This tutorial compares a naive baseline with a more expressive but still interpretable classifier.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```

Prepare the problem once so both learners face exactly the same data.

``` r
iris <- datasets::iris
slevels <- levels(iris$Species)

set_example_seed()
sr <- train_test(sample_stratified("Species"), iris)
iris_train <- sr$train
iris_test <- sr$test
```

Train the majority classifier. It always predicts the most frequent class seen during training, so it is intentionally weak but very informative as a minimum benchmark. In `daltoolbox`, classifier predictions are returned as class-score tables, and `evaluate()` can compare them directly with the original factor labels.

``` r
baseline <- cla_majority("Species", slevels)
set_example_seed()
baseline <- fit(baseline, iris_train)

baseline_pred <- predict(baseline, iris_test)
baseline_eval <- evaluate(baseline, iris_test$Species, baseline_pred)
baseline_eval$metrics
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity  f1
## 1 0.3333333 10  0 20  0 0.3333333      1           1           0 0.5
```

Now train a decision tree on the same split. Because the experimental conditions are unchanged, any improvement is easier to interpret.

``` r
tree <- cla_dtree("Species", slevels)
set_example_seed()
tree <- fit(tree, iris_train)

tree_pred <- predict(tree, iris_test)
tree_eval <- evaluate(tree, iris_test$Species, tree_pred)
tree_eval$metrics
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9333333 10 20  0  0         1      1           1           1  1
```

The practical lesson is simple: always compare a more elaborate model with a transparent baseline, not only with other elaborate models.
