About sampling
- `sample_random`: splits train/test sets and creates folds via random draws, preserving only expected proportions on average.

This example is especially useful for beginners because it shows that data mining quality starts before model fitting. A weak split can compromise even a good learner.

Didactic goal: interpret sampling as an experimental design choice, not as a routine preamble.

Environment setup.

``` r
# Sampling dataset

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

Load dataset and view class distribution.

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

Split into train/test with random sampling (distributions may vary by draw).

``` r
# Split into train and test

# using random sampling
tt <- train_test(sample_random(), iris)

# training distribution
print(table(tt$train$Species))
```

```
## 
##     setosa versicolor  virginica 
##         33         42         45
```

``` r
# test distribution
print(table(tt$test$Species))
```

```
## 
##     setosa versicolor  virginica 
##         17          8          5
```

What to observe
- The training and test class counts are close to the original distribution, but not guaranteed to match it exactly.
- This variability is the main reason random sampling can behave differently across repeated runs.

Create random k-folds and check distribution per fold.

``` r
# Split the dataset into folds

# preparing the dataset into four folds
sample <- sample_random()
folds <- k_fold(sample, iris, 4)

# folds distribution
tbl <- NULL
for (f in folds) {
tbl <- rbind(tbl, table(f$Species))
}
print(tbl)
```

```
##      setosa versicolor virginica
## [1,]     14         17         6
## [2,]     14          7        16
## [3,]     10         15        13
## [4,]     12         11        15
```

Common mistakes
- Assuming random sampling preserves class proportions perfectly in every split.
- Comparing models built on different random partitions without controlling the seed.
- Using random sampling in a clearly imbalanced problem when stratified sampling would be more stable.

References
- Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection. IJCAI.
