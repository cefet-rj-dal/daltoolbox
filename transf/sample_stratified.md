---
title: An R Markdown document converted from "Rmd/transf/sample_stratified.ipynb"
output: html_document
---

# Stratified sampling dataset


```r
# DAL ToolBox
# version 1.1.727

source("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/jupyter.R")

#loading DAL
load_library("daltoolbox") 
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
tt <- train_test(sample_stratified("Species"), iris)

# distribution of train
print(table(tt$train$Species))
```

```
## 
##     setosa versicolor  virginica 
##         40         40         40
```

```r
# distribution of test
print(table(tt$test$Species))
```

```
## 
##     setosa versicolor  virginica 
##         10         10         10
```

## Dividing a dataset into folds


```r
#using stratified sampling
# preparing dataset into four folds
sample <- sample_stratified("Species")
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
## [1,]     13         13        13
## [2,]     13         13        13
## [3,]     12         12        12
## [4,]     12         12        12
```

