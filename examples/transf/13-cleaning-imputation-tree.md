About the transformation
- `imputation_tree`: iterative tree-based imputation for mixed datasets.

Didactic goal: contrast model-based imputation with simpler fill rules such as mean or median replacement. The method uses other attributes to predict missing values, which usually makes the imputation more context-aware.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```

Create a dataset with missing values in both numeric and categorical columns.

``` r
iris_na <- datasets::iris
iris_na$Sepal.Length[c(2, 10, 25)] <- NA
iris_na$Petal.Width[c(5, 11)] <- NA
iris_na$Species[c(3, 15)] <- NA

summary(iris_na)
```

```
##   Sepal.Length    Sepal.Width     Petal.Length    Petal.Width          Species  
##  Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100   setosa    :48  
##  1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300   versicolor:50  
##  Median :5.800   Median :3.000   Median :4.350   Median :1.300   virginica :50  
##  Mean   :5.863   Mean   :3.057   Mean   :3.758   Mean   :1.213   NAs       : 2  
##  3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800                  
##  Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500                  
##  NAs    :3                                       NAs    :2
```


``` r
imp <- imputation_tree(maxit = 3)
```

```
## Error in `imputation_tree()`:
## ! could not find function "imputation_tree"
```

``` r
imp <- fit(imp, iris_na)
```

```
## Error:
## ! object 'imp' not found
```

``` r
iris_imp <- transform(imp, iris_na)
```

```
## Error:
## ! object 'imp' not found
```

``` r
summary(iris_imp$Sepal.Length)
```

```
## Error:
## ! object 'iris_imp' not found
```

``` r
table(iris_imp$Species, useNA = "ifany")
```

```
## Error:
## ! object 'iris_imp' not found
```

References
- Stekhoven, D. J., and Buhlmann, P. (2012). MissForest: non-parametric missing value imputation for mixed-type data.
