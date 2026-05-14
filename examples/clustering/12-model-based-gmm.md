About the method
- `cluster_gmm`: Gaussian mixture model clustering.

Didactic goal: keep the same clustering line of experiment and change only the clustering family to a probabilistic mixture model.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "mclust"))

library(daltoolbox)
library(mclust)
```

```
##                    __           __ 
##    ____ ___  _____/ /_  _______/ /_
##   / __ `__ \/ ___/ / / / / ___/ __/
##  / / / / / / /__/ / /_/ (__  ) /_  
## /_/ /_/ /_/\___/_/\__,_/____/\__/   version 6.1.2
## Type 'citation("mclust")' for citing this R package in publications.
```

Load data and separate predictors from the reference labels used only for interpretation.

``` r
iris <- datasets::iris
x <- iris[, 1:4]
ref <- iris$Species
head(x)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width
## 1          5.1         3.5          1.4         0.2
## 2          4.9         3.0          1.4         0.2
## 3          4.7         3.2          1.3         0.2
## 4          4.6         3.1          1.5         0.2
## 5          5.0         3.6          1.4         0.2
## 6          5.4         3.9          1.7         0.4
```

Model configuration.

``` r
model <- cluster_gmm(G = 3)
```

Fit the model and obtain cluster labels.

``` r
model <- fit(model, x)
```

```
## fitting ...
## 
  |                                                                                                                        
  |                                                                                                                  |   0%
  |                                                                                                                        
  |========                                                                                                          |   7%
  |                                                                                                                        
  |===============                                                                                                   |  13%
  |                                                                                                                        
  |=======================                                                                                           |  20%
  |                                                                                                                        
  |==============================                                                                                    |  27%
  |                                                                                                                        
  |======================================                                                                            |  33%
  |                                                                                                                        
  |==============================================                                                                    |  40%
  |                                                                                                                        
  |=====================================================                                                             |  47%
  |                                                                                                                        
  |=============================================================                                                     |  53%
  |                                                                                                                        
  |====================================================================                                              |  60%
  |                                                                                                                        
  |============================================================================                                      |  67%
  |                                                                                                                        
  |====================================================================================                              |  73%
  |                                                                                                                        
  |===========================================================================================                       |  80%
  |                                                                                                                        
  |===================================================================================================               |  87%
  |                                                                                                                        
  |==========================================================================================================        |  93%
  |                                                                                                                        
  |==================================================================================================================| 100%
```

``` r
clu <- cluster(model, x)
```

```
## Error in `cluster.default()`:
## ! only implemented for resamples objects
```

``` r
table(clu)
```

```
## Error:
## ! object 'clu' not found
```

Evaluate the partition.

``` r
eval <- evaluate(model, clu, ref)
```

```
## Error:
## ! object 'clu' not found
```

``` r
eval
```

```
## function (expr, envir = parent.frame(), enclos = if (is.list(envir) || 
##     is.pairlist(envir)) parent.frame() else baseenv()) 
## .Internal(eval(expr, envir, enclos))
## <bytecode: 0x57adf7f87c10>
## <environment: namespace:base>
```

References
- Fraley, C., and Raftery, A. E. (2002). Model-Based Clustering, Discriminant Analysis, and Density Estimation.
