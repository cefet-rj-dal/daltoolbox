## Feature Selection with Forward Stepwise Subset Search

Forward stepwise subset search starts with no predictors and adds the feature that most improves the current regression model. This implementation keeps the subset with the highest adjusted R-squared.


``` r
# installation
# install.packages("daltoolbox")

library(daltoolbox)
```


``` r
iris <- datasets::iris
```


``` r
if (requireNamespace("leaps", quietly = TRUE)) {
  fs <- feature_selection_fss("Sepal.Length")
  fs <- fit(fs, iris)

  print(fs$selected)
  print(fs$ranking)

  iris_fs <- transform(fs, iris)
  head(iris_fs)
}
```

```
## [1] "Sepal.Width"  "Petal.Length" "Petal.Width" 
##        feature score
## 1  Sepal.Width     1
## 2 Petal.Length     2
## 3  Petal.Width     3
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width
## 1          5.1         3.5          1.4         0.2
## 2          4.9         3.0          1.4         0.2
## 3          4.7         3.2          1.3         0.2
## 4          4.6         3.1          1.5         0.2
## 5          5.0         3.6          1.4         0.2
## 6          5.4         3.9          1.7         0.4
```

References
- Hastie, T., Tibshirani, R., and Friedman, J. (2009). The Elements of Statistical Learning.
