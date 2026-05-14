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

Model configuration. The evaluation lists are customized here for a didactic reason: `silhouette` is a compact way to discuss cohesion and separation in centroid-based partitions, `davies_bouldin` adds a second internal view based on cluster scatter and separation, and `entropy` plus `purity` make the external comparison with `iris$Species` easy to read. This is a deliberate simplification of the default evaluation set of `cluster_kmeans()`.

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

What to observe
- The labels in `ref` are not used during fitting; they only help interpret the partition afterward.
- This is the reference clustering workflow used by the other examples in this family.

References
- MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.
- Davies, D. L., and Bouldin, D. W. (1979). A cluster separation measure.
- Zhao, Y., and Karypis, G. (2001). Criterion functions for document clustering.
