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
```

```
## Error in `bal_subsampling()`:
## ! could not find function "bal_subsampling"
```

``` r
iris_bal <- transform(bal, iris_imb)
```

```
## Error:
## ! object 'bal' not found
```

``` r
table(iris_bal$Species)
```

```
## Error:
## ! object 'iris_bal' not found
```

``` r
head(iris_bal)
```

```
## Error:
## ! object 'iris_bal' not found
```
