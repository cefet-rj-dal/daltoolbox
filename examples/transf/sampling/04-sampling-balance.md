About the transformation
- `sample_balance`: class balancing by up-sampling or down-sampling through the sampling interface.

Didactic goal: compare this general balancing helper with the more explicit oversampling and subsampling examples. Here the emphasis is on class-count correction, not on synthetic data generation.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```

Create an imbalanced subset and then rebalance it.

``` r
iris_imb <- datasets::iris[c(1:50, 51:71, 101:111), ]
table(iris_imb$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         21         11
```


``` r
bal_down <- sample_balance("Species", method = "down")
iris_down <- transform(bal_down, iris_imb)
table(iris_down$Species)
```

```
## 
##     setosa versicolor  virginica 
##         11         11         11
```


``` r
bal_up <- sample_balance("Species", method = "up")
iris_up <- transform(bal_up, iris_imb)
table(iris_up$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         50         50
```
