About the method
- `cluster_pam`: Partitioning Around Medoids. Similar to k-means, but uses medoids instead of centroids.

Didactic goal: keep the same clustering line of experiment and change only the partitioning strategy.

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

Model configuration. The evaluation lists are customized here for a didactic reason: `silhouette` and `davies_bouldin` help compare the medoid partition through cohesion and separation, while `entropy` and `purity` make the external comparison with `iris$Species` easy to interpret. This keeps the discussion close to the k-means example so the reader can focus on the difference between centroids and medoids.

``` r
model <- cluster_pam(k = 3)
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
- The workflow is unchanged from k-means.
- The method-specific difference is that medoids are actual observations, which can make the partition more robust to outliers.

References
- Kaufman, L., and Rousseeuw, P. J. (1990). Finding Groups in Data.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.
- Davies, D. L., and Bouldin, D. W. (1979). A cluster separation measure.
- Zhao, Y., and Karypis, G. (2001). Criterion functions for document clustering.
