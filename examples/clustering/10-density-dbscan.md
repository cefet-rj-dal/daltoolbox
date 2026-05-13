About the method
- `cluster_dbscan`: density-based clustering that identifies dense regions and can label sparse points as noise.

Didactic goal: keep the same clustering line of experiment and change only the grouping principle, from partitioning all cases to detecting dense neighborhoods.

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
model <- cluster_dbscan(minPts = 3)
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
##  0  1  2  3  4 
## 26 47 38  4 35
```

Evaluate the partition.

``` r
eval <- evaluate(model, clu, ref)
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

What to observe
- The workflow is the same as in the partition-based methods.
- The method-specific difference is that DBSCAN may leave some cases as noise instead of forcing all observations into clusters.

References
- Ester, M., Kriegel, H.-P., Sander, J., and Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.
