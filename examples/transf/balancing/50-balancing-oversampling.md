## Random or SMOTE-Based Class Oversampling

This example balances minority classes either by random replication or by synthetic interpolation using the local SMOTE implementation built into `daltoolbox`.

Didactic goal: compare two ways of increasing minority-class representation without changing the modeling interface. The key point is not only that the class counts become more balanced, but also that the balancing strategy changes the type of synthetic information introduced into the training data.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation
# install.packages("daltoolbox")

library(daltoolbox)
```

Create an imbalanced version of `iris` so the effect of balancing becomes visible.

``` r
iris_imb <- datasets::iris[c(1:50, 51:71, 101:111), ]
table(iris_imb$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         21         11
```

Random oversampling duplicates existing minority cases until the classes become balanced.

``` r
set_example_seed()
bal_random <- bal_oversampling("Species", method = "random")
iris_random <- transform(bal_random, iris_imb)
table(iris_random$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         50         50
```

SMOTE creates synthetic minority instances by interpolation among nearby cases. This usually produces a richer balanced set than pure replication.

``` r
set_example_seed()
bal_smote <- bal_oversampling("Species", method = "smote", k = 3)
iris_smote <- transform(bal_smote, iris_imb)
table(iris_smote$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         50         50
```

``` r
head(iris_smote)
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

What to observe
- Random oversampling preserves original minority examples exactly, so duplicated rows may appear.
- SMOTE changes the dataset more deeply because it generates synthetic numeric combinations.
- Both approaches should be fitted only on the training split, not on the full dataset, to avoid leakage into evaluation.

References
- Chawla, N. V., Bowyer, K. W., Hall, L. O., and Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique.
