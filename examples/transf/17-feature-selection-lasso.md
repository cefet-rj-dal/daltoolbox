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
## Error in `feature_selection_lasso()`:
## ! could not find function "feature_selection_lasso"
```

References
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso.
