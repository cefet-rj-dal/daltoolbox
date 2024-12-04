# Sampling dataset


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
table(iris$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         50         50
```

## Dividing a dataset with training and test


```r
#using random sampling
tt <- train_test(sample_random(), iris)

# distribution of train
print(table(tt$train$Species))
```

```
## 
##     setosa versicolor  virginica 
##         39         39         42
```

```r
# distribution of test
print(table(tt$test$Species))
```

```
## 
##     setosa versicolor  virginica 
##         11         11          8
```

## Dividing a dataset into folds


```r
# preparing dataset into four folds
sample <- sample_random()
folds <- k_fold(sample, iris, 4)

# distribution of folds
tbl <- NULL
for (f in folds) {
tbl <- rbind(tbl, table(f$Species))
}
print(tbl)
```

```
##      setosa versicolor virginica
## [1,]     14         16         7
## [2,]     10         12        15
## [3,]     15         10        13
## [4,]     11         12        15
```

