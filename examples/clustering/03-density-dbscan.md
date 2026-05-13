About the method
- `cluster_dbscan`: density-based method. Identifies dense regions separated by sparse areas; detects noise and arbitrarily shaped clusters.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# Clustering - dbscan

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

Didactic goal: read this example as an unsupervised workflow. The emphasis is not on predicting a known label during training, but on understanding how the method groups the data and how preprocessing affects that grouping.

Load data (`iris`).

``` r
# loading dataset
data(iris)
```

Configure DBSCAN; tune `minPts` (and `eps` if available) according to density.

``` r
# clustering method configuration
model <- cluster_dbscan(minPts = 3)
model$eval_external <- list(
  model$clu_utils$metric_entropy,
  model$clu_utils$metric_purity
)
```

Fit and obtain cluster labels.

``` r
# model fitting and labeling
set_example_seed()
model <- fit(model, iris[,1:4])
clu <- cluster(model, iris[,1:4])
table(clu)
```

```
## clu
##  0  1  2  3  4 
## 26 47 38  4 35
```

External evaluation using `Species`, plus the internal count of noise points used by the default DBSCAN configuration.

``` r
# evaluate model using internal and external metrics
eval <- evaluate(model, clu, iris$Species)
eval
```

```
## $clusters_entropy
## # A tibble: 5 × 4
##   x        ce   qtd    ceg
##   <fct> <dbl> <int>  <dbl>
## 1 0     1.18     26 0.205 
## 2 1     0        47 0     
## 3 2     0        38 0     
## 4 3     0         4 0     
## 5 4     0.422    35 0.0985
## 
## $clustering_entropy
## [1] 0.3037218
## 
## $data_entropy
## [1] 1.584963
## 
## $metrics
##         metric      value     goal     type
## 1 noise_points 26.0000000 minimize internal
## 2      entropy  0.3037218 minimize external
## 3       purity  0.9266667 maximize external
```

References
- Ester, M., Kriegel, H.-P., Sander, J., Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.
