About the method
- `cla_knn`: k-Nearest Neighbors classifier. Classifies by majority vote among the k nearest neighbors in feature space.
- Main parameter: `k` (number of neighbors). Small k may overfit noise; larger k smooths the decision boundary.

Environment setup: install and load the package.

``` r
# Classification using KNN

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

Load sample data (iris) and initial inspection.

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

Identify target `Species` levels to configure the classifier.

``` r
# extracting the levels for the dataset
slevels <- levels(iris$Species)
slevels
```

```
## [1] "setosa"     "versicolor" "virginica"
```

Building train and test samples via random sampling
Train/test split with reproducible random sampling.

``` r
# Building train and test samples via random sampling
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test
```

Check class distribution for full dataset, training, and test.

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
## training     39         38        43
## test         11         12         7
```

Model training
Train the kNN model: set target, levels, and k.

``` r
# Model training
model <- cla_knn("Species", slevels, k=1) # k=1 for nearest neighbor
model <- fit(model, iris_train)
```

Training evaluation
Predict on training set, adjust labels, and compute metrics.

``` r
# Checking fit on training data
train_prediction <- predict(model, iris_train)

# Model evaluation (training)
iris_train_predictand <- adjust_class_label(iris_train[,"Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
print(train_eval$metrics)
```

```
##   accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1        1 39 81  0  0         1      1           1           1  1
```

Test evaluation
Predict on test set and compute metrics.

``` r
# Model test
test_prediction <- predict(model, iris_test)

iris_test_predictand <- adjust_class_label(iris_test[,"Species"])

# Test evaluation
 test_eval <- evaluate(model, iris_test_predictand, test_prediction)
print(test_eval$metrics)
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9333333 11 19  0  0         1      1           1           1  1
```

References
- Cover, T. and Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory.
