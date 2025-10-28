About the method
- `cluster_dbscan`: density-based method. Identifies dense regions separated by sparse areas; detects noise and arbitrarily shaped clusters.


``` r
# Clustering - dbscan

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

Load data (`iris`).

``` r
# loading dataset
data(iris)
```

Configure DBSCAN; tune `minPts` (and `eps` if available) according to density.

``` r
# clustering method configuration
model <- cluster_dbscan(minPts = 3)
```

Fit and obtain cluster labels.

``` r
# model fitting and labeling
model <- fit(model, iris[,1:4])
clu <- cluster(model, iris[,1:4])
table(clu)
```

```
## clu
##  0  1  2  3  4 
## 26 47 38  4 35
```

External evaluation using `Species` (note: DBSCAN may mark noise).

``` r
# evaluate model using external metric
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
```

References
- Ester, M., Kriegel, H.-P., Sander, J., Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.
