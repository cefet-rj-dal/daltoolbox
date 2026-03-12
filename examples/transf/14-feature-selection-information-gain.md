## Feature Selection with Information Gain

Information Gain measures the reduction in target entropy provided by each feature. This example ranks the predictors and keeps the most informative ones for a categorical target.


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
fs <- feature_selection_info_gain("Species", top = 2)
```

```
## Error in `feature_selection_info_gain()`:
## ! could not find function "feature_selection_info_gain"
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
- Quinlan, J. R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann.
