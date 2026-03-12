## Feature Selection with Relief

Relief estimates feature relevance by comparing each observation with its nearest hit and nearest miss. The method is useful for ranking predictors in classification problems.


``` r
# installation
# install.packages("daltoolbox")

library(daltoolbox)
```


``` r
iris <- datasets::iris
iris$Species <- factor(iris$Species)
```


``` r
fs <- feature_selection_relief("Species", top = 2, m = 50, seed = 1)
```

```
## Error in `feature_selection_relief()`:
## ! could not find function "feature_selection_relief"
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
- Kononenko, I. (1994). Estimating attributes: Analysis and extensions of Relief. ECML.
