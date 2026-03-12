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
fs <- fit(fs, iris)

print(fs$selected)
```

```
## [1] "Sepal.Width"  "Sepal.Length"
```

``` r
print(fs$ranking)
```

```
##        feature     score
## 1  Sepal.Width 0.3571324
## 2 Sepal.Length 0.0000000
## 3 Petal.Length 0.0000000
## 4  Petal.Width 0.0000000
```


``` r
iris_fs <- transform(fs, iris)
head(iris_fs)
```

```
##   Species Sepal.Width Sepal.Length
## 1  setosa         3.5          5.1
## 2  setosa         3.0          4.9
## 3  setosa         3.2          4.7
## 4  setosa         3.1          4.6
## 5  setosa         3.6          5.0
## 6  setosa         3.9          5.4
```

References
- Quinlan, J. R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann.
