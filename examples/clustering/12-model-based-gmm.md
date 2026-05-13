About the method
- `cluster_gmm`: Gaussian mixture model clustering.

Didactic goal: keep the same clustering line of experiment and change only the clustering family to a probabilistic mixture model.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "mclust"))

library(daltoolbox)
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
if (requireNamespace("mclust", quietly = TRUE)) {
  model <- cluster_gmm(G = 3)
}
```

Fit the model and obtain cluster labels.

``` r
if (requireNamespace("mclust", quietly = TRUE)) {
  model <- fit(model, x)
  clu <- cluster(model, x)
  table(clu)
}
```

```
## fitting ...
## 
  |                                                                                                                           
  |                                                                                                                     |   0%
  |                                                                                                                           
  |========                                                                                                             |   7%
  |                                                                                                                           
  |================                                                                                                     |  13%
  |                                                                                                                           
  |=======================                                                                                              |  20%
  |                                                                                                                           
  |===============================                                                                                      |  27%
  |                                                                                                                           
  |=======================================                                                                              |  33%
  |                                                                                                                           
  |===============================================                                                                      |  40%
  |                                                                                                                           
  |=======================================================                                                              |  47%
  |                                                                                                                           
  |==============================================================                                                       |  53%
  |                                                                                                                           
  |======================================================================                                               |  60%
  |                                                                                                                           
  |==============================================================================                                       |  67%
  |                                                                                                                           
  |======================================================================================                               |  73%
  |                                                                                                                           
  |==============================================================================================                       |  80%
  |                                                                                                                           
  |=====================================================================================================                |  87%
  |                                                                                                                           
  |=============================================================================================================        |  93%
  |                                                                                                                           
  |=====================================================================================================================| 100%
```

```
## clu
##  1  2  3 
## 50 45 55
```

Evaluate the partition.

``` r
if (requireNamespace("mclust", quietly = TRUE)) {
  eval <- evaluate(model, clu, ref)
  eval
}
```

```
## $clusters_entropy
## # A tibble: 3 × 4
##   x        ce   qtd   ceg
##   <fct> <dbl> <int> <dbl>
## 1 1     0        50 0    
## 2 2     0        45 0    
## 3 3     0.439    55 0.161
## 
## $clustering_entropy
## [1] 0.1611489
## 
## $data_entropy
## [1] 1.584963
## 
## $metrics
##                metric        value     goal     type
## 1              loglik -186.0740479 maximize    model
## 2             entropy    0.1611489 minimize external
## 3              purity    0.9666667 maximize external
## 4 adjusted_rand_index    0.9038742 maximize external
```

References
- Fraley, C., and Raftery, A. E. (2002). Model-Based Clustering, Discriminant Analysis, and Density Estimation.
