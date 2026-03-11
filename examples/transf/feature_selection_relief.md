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
fs <- fit(fs, iris)

print(fs$selected)
```

```
## [1] "Petal.Width"  "Petal.Length"
```

``` r
print(fs$ranking)
```

```
##        feature     score
## 1  Petal.Width 0.2108333
## 2 Petal.Length 0.1633898
## 3  Sepal.Width 0.1508333
## 4 Sepal.Length 0.0600000
```


``` r
iris_fs <- transform(fs, iris)
head(iris_fs)
```

```
##   Species Petal.Width Petal.Length
## 1  setosa         0.2          1.4
## 2  setosa         0.2          1.4
## 3  setosa         0.2          1.3
## 4  setosa         0.2          1.5
## 5  setosa         0.2          1.4
## 6  setosa         0.4          1.7
```

References
- Kononenko, I. (1994). Estimating attributes: Analysis and extensions of Relief. ECML.
