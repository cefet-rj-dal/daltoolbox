About the method
- `cluster_kmeans`: partitions data into k groups by minimizing within-cluster variance. Sensitive to scale; normalization can improve results.


``` r
# Clustering - Kmeans

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox)  
```

Load sample data (`iris`).

``` r
# loading dataset
data(iris)
```

Configure K-means with k=3 (one cluster per species in iris).

``` r
# clustering method configuration
model <- cluster_kmeans(k=3)
```

Fit the model and obtain cluster labels.

``` r
# model fitting and labeling
model <- fit(model, iris[,1:4])
clu <- cluster(model, iris[,1:4])
table(clu)
```

```
## clu
##  1  2  3 
## 96 33 21
```

External evaluation using true labels (`Species`).

``` r
# evaluate model using external metric
eval <- evaluate(model, clu, iris$Species)
eval
```

```
## $clusters_entropy
## # A tibble: 3 × 4
##   x        ce   qtd    ceg
##   <fct> <dbl> <int>  <dbl>
## 1 1     0.999    96 0.639 
## 2 2     0        33 0     
## 3 3     0.702    21 0.0983
## 
## $clustering_entropy
## [1] 0.7375436
## 
## $data_entropy
## [1] 1.584963
```


Effect of normalization: compare results after min-max.

``` r
# Influence of normalization in clustering

iris_minmax <- transform(fit(minmax(), iris), iris)
model <- fit(model, iris_minmax[,1:4])
clu <- cluster(model, iris_minmax[,1:4])
table(clu)
```

```
## clu
##  1  2  3 
## 39 50 61
```

Re-evaluation with normalized data.

``` r
# evaluate model using external metric

eval <- evaluate(model, clu, iris_minmax$Species)
eval
```

```
## $clusters_entropy
## # A tibble: 3 × 4
##   x        ce   qtd   ceg
##   <fct> <dbl> <int> <dbl>
## 1 1     0.391    39 0.102
## 2 2     0        50 0    
## 3 3     0.777    61 0.316
## 
## $clustering_entropy
## [1] 0.4177655
## 
## $data_entropy
## [1] 1.584963
```

References
- MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations.
- Lloyd, S. (1982). Least squares quantization in PCM.
