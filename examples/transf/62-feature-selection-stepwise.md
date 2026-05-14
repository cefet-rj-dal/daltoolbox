## Feature Selection with Stepwise Search

Stepwise search iteratively adds or removes predictors from a generalized linear model according to an information criterion. This example uses forward search for a regression target.


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
fs <- feature_selection_stepwise(
  "medv",
  direction = "forward",
  family = stats::gaussian
)
set_example_seed()
fs <- fit(fs, Boston)
```

```
print(fs$selected)
```

```
##  [1] "lstat"   "rm"      "ptratio" "dis"     "nox"     "chas"    "black"  
##  [8] "zn"      "crim"    "rad"     "tax"
```

``` r
print(fs$ranking)
```

```
##    feature score
## 1    lstat     1
## 2       rm     2
## 3  ptratio     3
## 4      dis     4
## 5      nox     5
## 6     chas     6
## 7    black     7
## 8       zn     8
## 9     crim     9
## 10     rad    10
## 11     tax    11
```


``` r
boston_fs <- transform(fs, Boston)
head(boston_fs)
```

```
##   medv lstat    rm ptratio    dis   nox chas  black zn    crim rad tax
## 1 24.0  4.98 6.575    15.3 4.0900 0.538    0 396.90 18 0.00632   1 296
## 2 21.6  9.14 6.421    17.8 4.9671 0.469    0 396.90  0 0.02731   2 242
## 3 34.7  4.03 7.185    17.8 4.9671 0.469    0 392.83  0 0.02729   2 242
## 4 33.4  2.94 6.998    18.7 6.0622 0.458    0 394.63  0 0.03237   3 222
## 5 36.2  5.33 7.147    18.7 6.0622 0.458    0 396.90  0 0.06905   3 222
## 6 28.7  5.21 6.430    18.7 6.0622 0.458    0 394.12  0 0.02985   3 222
```

References
- Hastie, T., Tibshirani, R., and Friedman, J. (2009). The Elements of Statistical Learning.
