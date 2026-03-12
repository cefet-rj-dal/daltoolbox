## Feature Selection with Stepwise Search

Stepwise search iteratively adds or removes predictors from a generalized linear model according to an information criterion. This example uses forward search for a binary target.


``` r
# installation
# install.packages("daltoolbox")

library(daltoolbox)
```


``` r
iris <- datasets::iris
iris$IsVersicolor <- factor(ifelse(iris$Species == "versicolor", "yes", "no"))
```


``` r
fs <- feature_selection_stepwise(
  "IsVersicolor",
  direction = "forward",
  family = stats::binomial
)
```

```
## Error in `feature_selection_stepwise()`:
## ! could not find function "feature_selection_stepwise"
```

``` r
fs <- fit(fs, iris)
```

```
## Error:
## ! object 'fs' not found
```

``` r
print(fs$selected)
```

```
## Error:
## ! object 'fs' not found
```

``` r
print(fs$ranking)
```

```
## Error:
## ! object 'fs' not found
```


``` r
iris_fs <- transform(fs, iris)
```

```
## Error:
## ! object 'fs' not found
```

``` r
head(iris_fs)
```

```
## Error:
## ! object 'iris_fs' not found
```

References
- Hastie, T., Tibshirani, R., and Friedman, J. (2009). The Elements of Statistical Learning.
