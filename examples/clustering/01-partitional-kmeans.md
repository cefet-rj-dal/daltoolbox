About the method
- `cluster_kmeans`: partitions data into `k` groups by minimizing within-cluster variance.

Didactic goal: establish the standard clustering line of experiment used throughout this family. Later examples should be read as changes in the clustering method, not as changes in the workflow.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

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
model <- cluster_kmeans(k = 3)
model$eval_internal <- list(
  model$clu_utils$metric_silhouette,
  model$clu_utils$metric_davies_bouldin
)
model$eval_external <- list(
  model$clu_utils$metric_entropy,
  model$clu_utils$metric_purity
)
```

Fit the model and obtain cluster labels.

``` r
set_example_seed()
model <- fit(model, x)
clu <- cluster(model, x)
table(clu)
```

```
## clu
##  1  2  3 
## 50 62 38
```

Evaluate the partition.

``` r
eval <- evaluate(model, clu, ref)
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

What to observe
- The labels in `ref` are not used during fitting; they only help interpret the partition afterward.
- This is the reference clustering workflow used by the other examples in this family.

References
- MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations.
