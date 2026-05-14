About the transformation
- `imputation_tree`: tree-based predictive imputation for one target column at a time.

Didactic goal: contrast model-based imputation with simpler fill rules such as mean or median replacement. The method uses other attributes to predict missing values in one target column, which keeps the API simple and makes the modeling role explicit.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```

Create a dataset with missing values in one target column.

``` r
iris_na <- datasets::iris
iris_na$Sepal.Length[c(2, 10, 25)] <- NA

summary(iris_na)
```

```
##   Sepal.Length    Sepal.Width     Petal.Length    Petal.Width          Species  
##  Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100   setosa    :50  
##  1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300   versicolor:50  
##  Median :5.800   Median :3.000   Median :4.350   Median :1.300   virginica :50  
##  Mean   :5.863   Mean   :3.057   Mean   :3.758   Mean   :1.199                  
##  3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800                  
##  Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500                  
##  NAs    :3
```


``` r
imp <- imputation_tree("Sepal.Length")
imp <- fit(imp, iris_na)
iris_imp <- transform(imp, iris_na)

summary(iris_imp$Sepal.Length)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##   4.300   5.100   5.800   5.843   6.400   7.900
```

``` r
sum(is.na(iris_imp$Sepal.Length))
```

```
## [1] 0
```

References
- Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984). Classification and Regression Trees.
- van Buuren, S., and Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R.
