## Custom Clustering

The primary goal of this example is to show how a clustering method can be customized while preserving the same Experiment Line logic used elsewhere in `daltoolbox`. The customization procedure is compact: define a constructor, implement `fit()` to store the learned structure, implement `cluster()` to produce labels, and keep the standard evaluation step available.

This is the practical value of the framework: even when the algorithm comes from outside the package, the workflow remains stable and easy to read. In this concrete example, the custom clusterer uses `cluster::agnes`, an agglomerative hierarchical clustering method.


``` r
# installation
# install.packages("daltoolbox")

library(daltoolbox)
```


``` r
cluster_agnes_custom <- function(k = 3, method = "ward", metric = "euclidean", scale = TRUE) {
  obj <- daltoolbox::clusterer()
  obj$k <- k
  obj$method <- method
  obj$metric <- metric
  obj$scale <- scale
  class(obj) <- append("cluster_agnes_custom", class(obj))
  obj
}

fit.cluster_agnes_custom <- function(obj, data, ...) {
  x <- as.matrix(data)
  storage.mode(x) <- "double"

  if (isTRUE(obj$scale)) {
    x <- scale(x)
  }

  obj$model <- cluster::agnes(x, diss = FALSE, metric = obj$metric, method = obj$method)
  obj
}

cluster.cluster_agnes_custom <- function(obj, data, ...) {
  if (is.null(obj$model)) {
    obj <- fit(obj, data)
  }

  x <- as.matrix(data)
  storage.mode(x) <- "double"

  clu <- stats::cutree(as.hclust(obj$model), k = obj$k)

  dist <- 0
  for (i in unique(clu)) {
    idx <- i == clu
    center <- colMeans(x[idx, , drop = FALSE])
    dist <- dist + sum(rowSums((x[idx, , drop = FALSE] - center)^2))
  }

  attr(clu, "metric") <- dist
  clu
}
```


``` r
iris <- datasets::iris
model <- cluster_agnes_custom(k = 3, method = "ward")
model <- fit(model, iris[, 1:4])
clu <- cluster(model, iris[, 1:4])
table(clu)
```

```
## clu
##  1  2  3 
## 49 30 71
```


``` r
eval <- evaluate(model, clu, iris$Species)
eval
```

```
## $clusters_entropy
## # A tibble: 3 × 4
##   x        ce   qtd   ceg
##   <fct> <dbl> <int> <dbl>
## 1 1     0        49 0    
## 2 2     0.561    30 0.112
## 3 3     0.909    71 0.430
## 
## $clustering_entropy
## [1] 0.5422445
## 
## $data_entropy
## [1] 1.584963
```

References
- Kaufman, L., and Rousseeuw, P. J. (1990). Finding Groups in Data: An Introduction to Cluster Analysis.
