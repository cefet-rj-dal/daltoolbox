## Tutorial 02 - Sampling Strategy

A good experiment starts with a good split. Before choosing a sophisticated algorithm, it is worth asking whether the training and test data are representative of the same problem. In classification, one common concern is class balance: if the target distribution changes too much across splits, evaluation becomes misleading.

This tutorial explains why stratified sampling is often preferable when the target is categorical.


``` r
# install.packages("daltoolbox")

library(daltoolbox)
```

First inspect the original class distribution. This gives the reference that the later splits should preserve as much as possible.

``` r
iris <- datasets::iris
table(iris$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         50         50
```

Next build a stratified train/test split. Stratification tries to keep the class proportions similar in both subsets.

``` r
set.seed(1)
tt <- train_test(sample_stratified("Species"), iris)

table(tt$train$Species)
```

```
## 
##     setosa versicolor  virginica 
##         40         40         40
```

``` r
table(tt$test$Species)
```

```
## 
##     setosa versicolor  virginica 
##         10         10         10
```

Many studies also need repeated validation, not just one split. The next block creates folds that preserve the same class proportions, which is useful for cross-validation style workflows.

``` r
sample <- sample_stratified("Species")
folds <- k_fold(sample, iris, 4)

tbl <- NULL
for (f in folds) {
  tbl <- rbind(tbl, table(f$Species))
}
tbl
```

```
##      setosa versicolor virginica
## [1,]     13         13        13
## [2,]     13         13        13
## [3,]     12         12        12
## [4,]     12         12        12
```

To connect sampling with modeling, fit a simple baseline on the stratified split. The goal here is not high performance; it is to show that even a basic learner depends on the quality of the sampling procedure.

``` r
slevels <- levels(iris$Species)

model <- cla_majority("Species", slevels)
model <- fit(model, tt$train)

prediction <- predict(model, tt$test)
eval <- evaluate(model, adjust_class_label(tt$test$Species), prediction)
eval$metrics
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity  f1
## 1 0.3333333 10  0 20  0 0.3333333      1           1           0 0.5
```

The key lesson is practical: before comparing models, make sure the data partition itself is not introducing unnecessary distortion.
