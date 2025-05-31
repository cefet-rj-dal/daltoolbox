
``` r
# Stratified sampling dataset

# installation 
install.packages("daltoobox")
```

```
## Installing package into '/home/gpca/R/x86_64-pc-linux-gnu-library/4.5'
## (as 'lib' is unspecified)
```

```
## Warning in install.packages :
##   package 'daltoobox' is not available for this version of R
## 
## A version of this package for your version of R might be available elsewhere,
## see the ideas at
## https://cran.r-project.org/doc/manuals/r-patched/R-admin.html#Installing-packages
```

``` r
# loading DAL
library(daltoolbox) 
```


``` r
iris <- datasets::iris
head(iris)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
## 1          5.1         3.5          1.4         0.2  setosa
## 2          4.9         3.0          1.4         0.2  setosa
## 3          4.7         3.2          1.3         0.2  setosa
## 4          4.6         3.1          1.5         0.2  setosa
## 5          5.0         3.6          1.4         0.2  setosa
## 6          5.4         3.9          1.7         0.4  setosa
```

``` r
table(iris$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         50         50
```


``` r
# Dividing a dataset with training and test

# using random sampling
tt <- train_test(sample_stratified("Species"), iris)

# distribution of train
print(table(tt$train$Species))
```

```
## 
##     setosa versicolor  virginica 
##         40         40         40
```

``` r
# distribution of test
print(table(tt$test$Species))
```

```
## 
##     setosa versicolor  virginica 
##         10         10         10
```


``` r
# Dividing a dataset into folds

# using stratified sampling
# preparing dataset into four folds
sample <- sample_stratified("Species")
folds <- k_fold(sample, iris, 4)

# distribution of folds
tbl <- NULL
for (f in folds) {
tbl <- rbind(tbl, table(f$Species))
}
print(tbl)
```

```
##      setosa versicolor virginica
## [1,]     13         13        13
## [2,]     13         13        13
## [3,]     12         12        12
## [4,]     12         12        12
```

