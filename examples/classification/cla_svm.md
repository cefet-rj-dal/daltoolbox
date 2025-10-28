About the method
- `cla_svm`: Support Vector Machine for classification, maximizing the margin between classes.
- Common hyperparameters: `cost` (penalty), `epsilon` (insensitive-margin width), and `kernel` (e.g., linear, radial, polynomial, sigmoid).

Environment setup: install and load the package.

``` r
# Classification using Support Vector Machine

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

Identify the target `Species` levels.

``` r
# extracting the levels for the dataset
slevels <- levels(iris$Species)
slevels
```

```
## [1] "setosa"     "versicolor" "virginica"
```

# Building train and test samples via random sampling
Random train/test split.

``` r
# Building train and test samples via random sampling
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test
```

Check class distribution after the split.

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

# Model training
Train SVM: tune `cost`, `epsilon`, and optionally `kernel`.

``` r
# Model training
model <- cla_svm("Species", slevels, epsilon=0.0, cost=20.000) # default kernel; adjust as needed
model <- fit(model, iris_train)
```

# Training evaluation
Predict and compute metrics.

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
## 1    0.975 39 81  0  0         1      1           1           1  1
```

# Test evaluation
Predict and compute metrics.

``` r
# Model test
test_prediction <- predict(model, iris_test)

iris_test_predictand <- adjust_class_label(iris_test[,"Species"])

# Test evaluation
 test_eval <- evaluate(model, iris_test_predictand, test_prediction)
print(test_eval$metrics)
```

```
##   accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1        1 11 19  0  0         1      1           1           1  1
```

References
- Cortes, C. and Vapnik, V. (1995). Support-Vector Networks. Machine Learning 20(3):273–297.
- Chang, C.-C. and Lin, C.-J. (2011). LIBSVM: A library for support vector machines.
