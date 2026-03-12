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
```

```
## Error in `bal_oversampling()`:
## ! could not find function "bal_oversampling"
```

``` r
iris_random <- transform(bal_random, iris_imb)
```

```
## Error:
## ! object 'bal_random' not found
```

``` r
table(iris_random$Species)
```

```
## Error:
## ! object 'iris_random' not found
```


``` r
bal_smote <- bal_oversampling("Species", method = "smote", k = 3, seed = 123)
```

```
## Error in `bal_oversampling()`:
## ! could not find function "bal_oversampling"
```

``` r
iris_smote <- transform(bal_smote, iris_imb)
```

```
## Error:
## ! object 'bal_smote' not found
```

``` r
table(iris_smote$Species)
```

```
## Error:
## ! object 'iris_smote' not found
```

``` r
head(iris_smote)
```

```
## Error:
## ! object 'iris_smote' not found
```
