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
fs <- feature_selection_stepwise(
  "medv",
  direction = "forward",
  family = stats::gaussian
)
set_example_seed()
fs <- fit(fs, Boston)

print(fs$selected)
```

```
##  [1] "lstat"   "rm"      "ptratio" "dis"     "nox"     "chas"    "black"   "zn"      "crim"    "rad"     "tax"
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
