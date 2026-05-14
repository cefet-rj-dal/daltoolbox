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


``` r
if (requireNamespace("glmnet", quietly = TRUE)) {
  fs <- feature_selection_lasso("medv")
  set_example_seed()
  fs <- fit(fs, Boston)

  print(fs$selected)
  print(fs$ranking)

  boston_fs <- transform(fs, Boston)
  head(boston_fs)
}
```

```
##  [1] "crim"    "zn"      "chas"    "nox"     "rm"      "dis"     "rad"    
##  [8] "tax"     "ptratio" "black"   "lstat"  
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

```
##   medv  crim zn chas   nox    rm   dis rad tax ptratio  black lstat
## 1 24.0 0.006 18    0 0.538 6.575 4.090   1 296    15.3 396.90  4.98
## 2 21.6 0.027  0    0 0.469 6.421 4.967   2 242    17.8 396.90  9.14
## 3 34.7 0.027  0    0 0.469 7.185 4.967   2 242    17.8 392.83  4.03
## 4 33.4 0.032  0    0 0.458 6.998 6.062   3 222    18.7 394.63  2.94
## 5 36.2 0.069  0    0 0.458 7.147 6.062   3 222    18.7 396.90  5.33
## 6 28.7 0.030  0    0 0.458 6.430 6.062   3 222    18.7 394.12  5.21
```

References
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso.
