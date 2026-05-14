## Feature Selection with Forward Stepwise Subset Search

Forward stepwise subset search starts with no predictors and adds the feature that most improves the current regression model. This implementation keeps the subset with the highest adjusted R-squared.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation
# install.packages("daltoolbox")

library(daltoolbox)
```


``` r
data(Boston)
head(Boston)
```

```
##      crim zn indus chas   nox    rm  age    dis rad tax ptratio  black lstat medv
## 1 0.00632 18  2.31    0 0.538 6.575 65.2 4.0900   1 296    15.3 396.90  4.98 24.0
## 2 0.02731  0  7.07    0 0.469 6.421 78.9 4.9671   2 242    17.8 396.90  9.14 21.6
## 3 0.02729  0  7.07    0 0.469 7.185 61.1 4.9671   2 242    17.8 392.83  4.03 34.7
## 4 0.03237  0  2.18    0 0.458 6.998 45.8 6.0622   3 222    18.7 394.63  2.94 33.4
## 5 0.06905  0  2.18    0 0.458 7.147 54.2 6.0622   3 222    18.7 396.90  5.33 36.2
## 6 0.02985  0  2.18    0 0.458 6.430 58.7 6.0622   3 222    18.7 394.12  5.21 28.7
```


``` r
fs <- feature_selection_fss("medv")
set_example_seed()
fs <- fit(fs, Boston)

print(fs$selected)
```

```
##  [1] "crim"    "zn"      "chas"    "nox"     "rm"      "dis"     "rad"     "tax"     "ptratio" "black"   "lstat"
```

``` r
print(fs$ranking)
```

```
##    feature score
## 1     crim     1
## 2       zn     2
## 3     chas     3
## 4      nox     4
## 5       rm     5
## 6      dis     6
## 7      rad     7
## 8      tax     8
## 9  ptratio     9
## 10   black    10
## 11   lstat    11
```

``` r
boston_fs <- transform(fs, Boston)
head(boston_fs)
```

```
##   medv    crim zn chas   nox    rm    dis rad tax ptratio  black lstat
## 1 24.0 0.00632 18    0 0.538 6.575 4.0900   1 296    15.3 396.90  4.98
## 2 21.6 0.02731  0    0 0.469 6.421 4.9671   2 242    17.8 396.90  9.14
## 3 34.7 0.02729  0    0 0.469 7.185 4.9671   2 242    17.8 392.83  4.03
## 4 33.4 0.03237  0    0 0.458 6.998 6.0622   3 222    18.7 394.63  2.94
## 5 36.2 0.06905  0    0 0.458 7.147 6.0622   3 222    18.7 396.90  5.33
## 6 28.7 0.02985  0    0 0.458 6.430 6.0622   3 222    18.7 394.12  5.21
```

References
- Hastie, T., Tibshirani, R., and Friedman, J. (2009). The Elements of Statistical Learning.
