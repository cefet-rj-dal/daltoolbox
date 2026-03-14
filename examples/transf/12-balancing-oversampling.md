## Random or SMOTE-Based Class Oversampling

This example balances minority classes either by random replication or by synthetic interpolation using the local SMOTE implementation built into `daltoolbox`.


``` r
# installation
# install.packages("daltoolbox")

library(daltoolbox)
```


``` r
iris_imb <- datasets::iris[c(1:50, 51:71, 101:111), ]
table(iris_imb$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         21         11
```


``` r
bal_random <- bal_oversampling("Species", method = "random", seed = 123)
iris_random <- transform(bal_random, iris_imb)
table(iris_random$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         50         50
```


``` r
bal_smote <- bal_oversampling("Species", method = "smote", k = 3, seed = 123)
iris_smote <- transform(bal_smote, iris_imb)
table(iris_smote$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         50         50
```

``` r
head(iris_smote)
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
