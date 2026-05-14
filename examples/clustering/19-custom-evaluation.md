About the utility
- Clustering models in `daltoolbox` carry their own default internal and external evaluation metrics.
- Those defaults come from the class constructor, and they can be replaced when the analyst wants a narrower or different evaluation view.

Didactic goal: show that metric customization is optional, inspect the defaults of a clustering class, and then replace them explicitly in a controlled example.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```

Load data and keep the reference labels only for external interpretation.

``` r
iris <- datasets::iris
x <- iris[, 1:4]
ref <- iris$Species
```

Create a helper to list the metric names currently configured in a model.

``` r
metric_names <- function(model, x, clu, ref = NULL) {
  internal <- vapply(model$eval_internal, function(fn) {
    fn(data = x, cluster = clu, obj = model)$metric
  }, character(1))

  external <- if (is.null(ref)) {
    character(0)
  } else {
    vapply(model$eval_external, function(fn) {
      fn(cluster = clu, attribute = ref, obj = model)$metric
    }, character(1))
  }

  list(internal = internal, external = external)
}
```

Fit `cluster_kmeans()` and inspect its default evaluation set.

``` r
model_default <- cluster_kmeans(k = 3)
set_example_seed()
model_default <- fit(model_default, x)
clu_default <- cluster(model_default, x)
```

```
## Error in `cluster.default()`:
## ! only implemented for resamples objects
```

``` r
metric_names(model_default, x, clu_default, ref)
```

```
## Error:
## ! object 'clu_default' not found
```

Run the default evaluation.

``` r
eval_default <- evaluate(model_default, clu_default, ref)
```

```
## Error:
## ! object 'clu_default' not found
```

``` r
eval_default$metrics
```

```
## Error:
## ! object 'eval_default' not found
```

Now customize the same class to keep only one internal metric and one external metric. This pair is useful for teaching because `silhouette` gives a compact internal summary of cohesion and separation, while `entropy` shows how mixed each cluster is relative to a reference partition.

``` r
model_custom <- cluster_kmeans(k = 3)
model_custom$eval_internal <- list(model_custom$clu_utils$metric_silhouette)
model_custom$eval_external <- list(model_custom$clu_utils$metric_entropy)

set_example_seed()
model_custom <- fit(model_custom, x)
clu_custom <- cluster(model_custom, x)
```

```
## Error in `cluster.default()`:
## ! only implemented for resamples objects
```

``` r
metric_names(model_custom, x, clu_custom, ref)
```

```
## Error:
## ! object 'clu_custom' not found
```

Evaluate again after the customization.

``` r
eval_custom <- evaluate(model_custom, clu_custom, ref)
```

```
## Error:
## ! object 'clu_custom' not found
```

``` r
eval_custom$metrics
```

```
## Error:
## ! object 'eval_custom' not found
```

Available metric functions can be inspected directly from the utility object.

``` r
grep("^metric_", names(model_custom$clu_utils), value = TRUE)
```

```
##  [1] "metric_result"              "metric_from_attr"           "metric_wcss"                "metric_silhouette"         
##  [5] "metric_davies_bouldin"      "metric_calinski_harabasz"   "metric_entropy"             "metric_purity"             
##  [9] "metric_rand_index"          "metric_adjusted_rand_index" "metric_noise_points"        "metric_loglik"             
## [13] "metric_withinerror"         "metric_modularity"
```

What to observe
- The workflow does not change: configure, fit, cluster, and evaluate.
- What changes is only the content of `eval_internal` and `eval_external`.
- The default metric set is class-specific. The base `clusterer()` defines common defaults, and concrete classes can override them with more suitable metrics.

References
- MacQueen, J. (1967). Some Methods for classification and Analysis of Multivariate Observations.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.
- Shannon, C. E. (1948). A mathematical theory of communication.
