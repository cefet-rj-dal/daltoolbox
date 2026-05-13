About the method
- `cluster_pam`: Partitioning Around Medoids. Similar to k-means but uses medoids (real points) instead of centroids, making it more robust to outliers.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# Clustering - pam

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

Configure PAM with k=3.

``` r
# clustering method configuration
model <- cluster_pam(k=3)
model$eval_internal <- list(
  model$clu_utils$metric_silhouette,
  model$clu_utils$metric_davies_bouldin
)
model$eval_external <- list(
  model$clu_utils$metric_entropy,
  model$clu_utils$metric_purity
)
```

Fit and generate cluster labels.

``` r
# model fitting and labeling
set_example_seed()
model <- fit(model, iris[,1:4])
clu <- cluster(model, iris[,1:4])
table(clu)
```

```
## clu
##  1  2  3 
## 50 62 38
```

Internal and external evaluation.

``` r
# evaluate model using internal and external metrics
eval <- evaluate(model, clu, iris$Species)
eval
```

```
## $clusters_entropy
## # A tibble: 3 × 4
##   x        ce   qtd    ceg
##   <fct> <dbl> <int>  <dbl>
## 1 1     0        50 0     
## 2 2     0.771    62 0.319 
## 3 3     0.297    38 0.0754
## 
## $clustering_entropy
## [1] 0.3938863
## 
## $data_entropy
## [1] 1.584963
## 
## $metrics
##           metric     value     goal     type
## 1     silhouette 0.5528190 maximize internal
## 2 davies_bouldin 0.6619715 minimize internal
## 3        entropy 0.3938863 minimize external
## 4         purity 0.8933333 maximize external
```

References
- Kaufman, L. and Rousseeuw, P. J. (1990). Finding Groups in Data: An Introduction to Cluster Analysis.
