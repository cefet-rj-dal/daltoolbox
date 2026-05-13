About the chart
- `plot_pair`: scatter-matrix view of several numeric variables.

Didactic goal: move from one pair of variables to a compact multivariate inspection. This kind of chart is useful for exploratory pattern reading before more formal modeling.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "GGally"))

library(daltoolbox)
```


``` r
if (requireNamespace("GGally", quietly = TRUE)) {
  grf <- plot_pair(
    datasets::iris,
    cnames = colnames(datasets::iris)[1:4],
    title = "Iris scatter matrix",
    clabel = "Species"
  )
  print(grf)
}
```

```
## plot: [1, 1] [=====>-------------------------------------------------------------------------------------------] 6% est: 0s
```

```
## Warning: Use of `data[[clabel]]` is discouraged.
## ℹ Use `.data[[clabel]]` instead.
```

```
## plot: [1, 2] [===========>-------------------------------------------------------------------------------------] 12% est: 1s
## plot: [1, 3] [=================>-------------------------------------------------------------------------------] 19% est: 1s
## plot: [1, 4] [=======================>-------------------------------------------------------------------------] 25% est: 1s
## plot: [2, 1] [=============================>-------------------------------------------------------------------] 31% est: 1s
```

```
## Warning: Use of `data[[clabel]]` is discouraged.
## ℹ Use `.data[[clabel]]` instead.
```

```
## plot: [2, 2] [===================================>-------------------------------------------------------------] 38% est: 1s
```

```
## Warning: Use of `data[[clabel]]` is discouraged.
## ℹ Use `.data[[clabel]]` instead.
```

```
## plot: [2, 3] [=========================================>-------------------------------------------------------] 44% est: 1s
## plot: [2, 4] [===============================================>-------------------------------------------------] 50% est: 1s
## plot: [3, 1] [======================================================>------------------------------------------] 56% est: 1s
```

```
## Warning: Use of `data[[clabel]]` is discouraged.
## ℹ Use `.data[[clabel]]` instead.
```

```
## plot: [3, 2] [============================================================>------------------------------------] 62% est: 1s
```

```
## Warning: Use of `data[[clabel]]` is discouraged.
## ℹ Use `.data[[clabel]]` instead.
```

```
## plot: [3, 3] [==================================================================>------------------------------] 69% est: 1s
```

```
## Warning: Use of `data[[clabel]]` is discouraged.
## ℹ Use `.data[[clabel]]` instead.
```

```
## plot: [3, 4] [========================================================================>------------------------] 75% est: 0s
## plot: [4, 1] [==============================================================================>------------------] 81% est: 0s
```

```
## Warning: Use of `data[[clabel]]` is discouraged.
## ℹ Use `.data[[clabel]]` instead.
```

```
## plot: [4, 2] [====================================================================================>------------] 88% est: 0s
```

```
## Warning: Use of `data[[clabel]]` is discouraged.
## ℹ Use `.data[[clabel]]` instead.
```

```
## plot: [4, 3] [==========================================================================================>------] 94% est: 0s
```

```
## Warning: Use of `data[[clabel]]` is discouraged.
## ℹ Use `.data[[clabel]]` instead.
```

```
## plot: [4, 4] [=================================================================================================]100% est: 0s
```

```
## Warning: Use of `data[[clabel]]` is discouraged.
## ℹ Use `.data[[clabel]]` instead.
```

![plot of chunk unnamed-chunk-2](fig/23-relationship-pair/unnamed-chunk-2-1.png)
