## Random Class Undersampling

Random undersampling reduces all classes to the minority count by sampling without replacement.

Didactic goal: show the fastest way to reduce class imbalance and highlight its tradeoff. Undersampling is simple and often effective, but it discards information from the majority class to achieve balance.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation
# install.packages("daltoolbox")

library(daltoolbox)
```

Create an imbalanced dataset so the class reduction effect is easy to inspect.

``` r
iris_imb <- datasets::iris[c(1:50, 51:71, 101:111), ]
table(iris_imb$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         21         11
```

Apply undersampling and inspect the new class distribution.

``` r
set_example_seed()
bal <- bal_subsampling("Species")
iris_bal <- transform(bal, iris_imb)
table(iris_bal$Species)
```

```
## 
##     setosa versicolor  virginica 
##         11         11         11
```

``` r
head(iris_bal)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width   Species
## 1          6.3         3.3          6.0         2.5 virginica
## 2          6.5         3.0          5.8         2.2 virginica
## 3          6.5         3.2          5.1         2.0 virginica
## 4          6.7         2.5          5.8         1.8 virginica
## 5          5.8         2.7          5.1         1.9 virginica
## 6          6.3         2.9          5.6         1.8 virginica
```

What to observe
- The balanced result is obtained by removing majority-class rows, not by creating new minority rows.
- This makes undersampling computationally cheap, but potentially more lossy than oversampling.
- As with any balancing transformation, apply it to training data only.

References
- He, H., and Garcia, E. A. (2009). Learning from Imbalanced Data. IEEE Transactions on Knowledge and Data Engineering.
