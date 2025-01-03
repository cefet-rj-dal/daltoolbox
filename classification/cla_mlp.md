## Classification using Multilayer Perceptron


```r
# DAL ToolBox
# version 1.1.727



#loading DAL
library(daltoolbox) 
```


```r
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


```r
#extracting the levels for the dataset
slevels <- levels(iris$Species)
slevels
```

```
## [1] "setosa"     "versicolor" "virginica"
```


```r
#for performance issues, you can use matrix instead of data.frame (uncomment next line to test)
iris <- cbind(as.matrix(iris[,1:4]), Species=iris$Species)
```

## Building samples (training and testing)


```r
# preparing dataset for random sampling
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test
```


```r
tbl <- rbind(table(iris[,"Species"]), 
             table(iris_train[,"Species"]), 
             table(iris_test[,"Species"]))
rownames(tbl) <- c("dataset", "training", "test")
head(tbl)
```

```
##           1  2  3
## dataset  50 50 50
## training 39 38 43
## test     11 12  7
```

### Model training


```r
model <- cla_mlp("Species", slevels, size=3,decay=0.03)
model <- fit(model, iris_train)
train_prediction <- predict(model, iris_train)
```

### Model adjustment


```r
iris_train_predictand <- adjust_class_label(iris_train[,"Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
print(train_eval$metrics)
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9833333 39 81  0  0         1      1           1           1  1
```

### Model testing


```r
# Test  
test_prediction <- predict(model, iris_test)
  
iris_test_predictand <- adjust_class_label(iris_test[,"Species"])
test_eval <- evaluate(model, iris_test_predictand, test_prediction)
print(test_eval$metrics)
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9666667 11 19  0  0         1      1           1           1  1
```

