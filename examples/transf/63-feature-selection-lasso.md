## Feature Selection with Lasso

Lasso performs regression with an L1 penalty, shrinking some coefficients exactly to zero. This example keeps only the predictors with non-zero coefficients at the selected regularization level.


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
fs <- feature_selection_lasso("medv")
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
##         feature        score
## nox         nox 16.562664331
## rm           rm  3.851646315
## chas       chas  2.693903097
## dis         dis  1.419168850
## ptratio ptratio  0.933927773
## lstat     lstat  0.522521473
## rad         rad  0.263725830
## crim       crim  0.100714832
## zn           zn  0.042486737
## tax         tax  0.010286456
## black     black  0.009089735
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
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso.
