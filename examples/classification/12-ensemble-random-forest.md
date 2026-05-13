About the method
- `cla_rf`: Random Forest for classification. Ensemble of decision trees trained with randomness; robust and handles heterogeneous features well.
- Hyperparameters: `mtry` (variables per split), `ntree` (number of trees).

Didactic goal: read this example as a complete supervised-learning cycle. Pay attention not only to the learner call, but also to how the target is identified, how the split is created, and how training and test results should be interpreted separately.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# Classification using  Random Forest

# installation 
#install.packages("daltoolbox")

# loading DAL
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

Target `Species` levels.

``` r
# extracting the levels for the dataset
slevels <- levels(iris$Species)
slevels
```

```
## [1] "setosa"     "versicolor" "virginica"
```

Building train and test samples via random sampling
Random train/test split.

``` r
# Building train and test samples via random sampling
set_example_seed()
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test
```

Class distribution by split.

``` r
tbl <- rbind(table(iris[,"Species"]), 
             table(iris_train[,"Species"]), 
             table(iris_test[,"Species"]))
rownames(tbl) <- c("dataset", "training", "test")
head(tbl)
```

```
##          setosa versicolor virginica
## dataset      50         50        50
## training     41         39        40
## test          9         11        10
```

Model training
Train Random Forest: tune `mtry` and `ntree`.

``` r
# Model training
model <- cla_rf("Species", slevels, mtry=3, ntree=5) # mtry: variables per split; ntree: number of trees
set_example_seed()
model <- fit(model, iris_train)
```

Training evaluation

``` r
# Checking fit on training data
train_prediction <- predict(model, iris_train)

# Model evaluation (training)
train_eval <- evaluate(model, iris_train[,"Species"], train_prediction)
print(train_eval$metrics)
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9916667 41 79  0  0         1      1           1           1  1
```

Test evaluation

``` r
# Model test
test_prediction <- predict(model, iris_test)

# Test evaluation
 test_eval <- evaluate(model, iris_test[,"Species"], test_prediction)
print(test_eval$metrics)
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9333333  9 21  0  0         1      1           1           1  1
```

References
- Breiman, L. (2001). Random Forests. Machine Learning 45(1):5–32.
- Liaw, A. and Wiener, M. (2002). Classification and Regression by randomForest. R News.
