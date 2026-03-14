## Random Class Undersampling

Random undersampling reduces all classes to the minority count by sampling without replacement.


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
bal <- bal_subsampling("Species", seed = 123)
iris_bal <- transform(bal, iris_imb)
table(iris_bal$Species)
```

```
## 
##     setosa versicolor  virginica 
##         11         11         11
```

``` r
head(iris_bal)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width   Species
## 1          7.1         3.0          5.9         2.1 virginica
## 2          6.5         3.2          5.1         2.0 virginica
## 3          5.8         2.7          5.1         1.9 virginica
## 4          7.6         3.0          6.6         2.1 virginica
## 5          7.2         3.6          6.1         2.5 virginica
## 6          6.5         3.0          5.8         2.2 virginica
```
