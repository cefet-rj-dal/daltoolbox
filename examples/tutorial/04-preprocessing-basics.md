## Tutorial 04 - Preprocessing Basics

After the initial cleaning stage, the next question is how to make the attributes more suitable for modeling. Two common operations are scaling numeric variables and reducing redundancy among predictors.

This tutorial introduces both ideas in a compact way: normalization changes the scale, while correlation-based selection simplifies the feature set.


``` r
# install.packages("daltoolbox")

library(daltoolbox)
```

Begin by inspecting the original scale of the `iris` attributes. Notice that each numeric column has a different range.

``` r
iris <- datasets::iris
summary(iris)
```

```
##   Sepal.Length    Sepal.Width     Petal.Length    Petal.Width          Species  
##  Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100   setosa    :50  
##  1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300   versicolor:50  
##  Median :5.800   Median :3.000   Median :4.350   Median :1.300   virginica :50  
##  Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199                  
##  3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800                  
##  Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500
```

Apply min-max normalization so that all numeric attributes are placed on a comparable scale. This is especially important for methods that are sensitive to distances or magnitudes.

``` r
norm <- minmax()
norm <- fit(norm, iris)
iris_norm <- transform(norm, iris)

summary(iris_norm)
```

```
##   Sepal.Length     Sepal.Width      Petal.Length     Petal.Width            Species  
##  Min.   :0.0000   Min.   :0.0000   Min.   :0.0000   Min.   :0.00000   setosa    :50  
##  1st Qu.:0.2222   1st Qu.:0.3333   1st Qu.:0.1017   1st Qu.:0.08333   versicolor:50  
##  Median :0.4167   Median :0.4167   Median :0.5678   Median :0.50000   virginica :50  
##  Mean   :0.4287   Mean   :0.4406   Mean   :0.4675   Mean   :0.45806                  
##  3rd Qu.:0.5833   3rd Qu.:0.5417   3rd Qu.:0.6949   3rd Qu.:0.70833                  
##  Max.   :1.0000   Max.   :1.0000   Max.   :1.0000   Max.   :1.00000
```

Once the variables are normalized, check whether some attributes are strongly redundant. Correlation-based feature selection can help simplify the dataset without manually inspecting each pair of variables.

``` r
fs <- feature_selection_corr(cutoff = 0.9)
fs <- fit(fs, iris_norm)
```

Transform the dataset and compare the set of columns before and after selection. This shows the practical effect of the preprocessing step.

``` r
iris_fs <- transform(fs, iris_norm)

names(iris_norm)
```

```
## [1] "Sepal.Length" "Sepal.Width"  "Petal.Length" "Petal.Width"  "Species"
```

``` r
names(iris_fs)
```

```
## [1] "Species"      "Sepal.Length" "Sepal.Width"  "Petal.Width"
```

``` r
head(iris_fs)
```

```
##   Species Sepal.Length Sepal.Width Petal.Width
## 1  setosa   0.22222222   0.6250000  0.04166667
## 2  setosa   0.16666667   0.4166667  0.04166667
## 3  setosa   0.11111111   0.5000000  0.04166667
## 4  setosa   0.08333333   0.4583333  0.04166667
## 5  setosa   0.19444444   0.6666667  0.04166667
## 6  setosa   0.30555556   0.7916667  0.12500000
```

A beginner-friendly interpretation is: normalization prepares the scale, and feature selection prepares the structure of the data.
