## Feature Selection with Stepwise Search

Stepwise search iteratively adds or removes predictors from a generalized linear model according to an information criterion. This example uses forward search for a binary target.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
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
set_example_seed()
fs <- fit(fs, iris)
```

```
## Warning: glm.fit: algorithm did not converge
## Warning: glm.fit: algorithm did not converge
## Warning: glm.fit: algorithm did not converge
## Warning: glm.fit: algorithm did not converge
## Warning: glm.fit: algorithm did not converge
## Warning: glm.fit: algorithm did not converge
## Warning: glm.fit: algorithm did not converge
## Warning: glm.fit: algorithm did not converge
```

``` r
print(fs$selected)
```

```
## [1] "Species"
```

``` r
print(fs$ranking)
```

```
##   feature score
## 1 Species     1
```


``` r
iris_fs <- transform(fs, iris)
head(iris_fs)
```

```
##   IsVersicolor Species
## 1           no  setosa
## 2           no  setosa
## 3           no  setosa
## 4           no  setosa
## 5           no  setosa
## 6           no  setosa
```

References
- Hastie, T., Tibshirani, R., and Friedman, J. (2009). The Elements of Statistical Learning.
