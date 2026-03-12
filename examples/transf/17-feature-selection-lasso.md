## Feature Selection with Lasso

Lasso performs regression with an L1 penalty, shrinking some coefficients exactly to zero. This example keeps only the predictors with non-zero coefficients at the selected regularization level.


``` r
# installation
# install.packages("daltoolbox")

library(daltoolbox)
```


``` r
iris <- datasets::iris
```


``` r
if (requireNamespace("glmnet", quietly = TRUE)) {
  fs <- feature_selection_lasso("Sepal.Length")
  fs <- fit(fs, iris)

  print(fs$selected)
  print(fs$ranking)

  iris_fs <- transform(fs, iris)
  head(iris_fs)
}
```

```
## [1] "Sepal.Width"  "Petal.Length" "Petal.Width" 
##                   feature     score
## Petal.Length Petal.Length 0.6967400
## Sepal.Width   Sepal.Width 0.6453987
## Petal.Width   Petal.Width 0.5291831
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
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso.
