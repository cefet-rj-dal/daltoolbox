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
model_default <- daltoolbox::fit(model_default, x)
clu_default <- daltoolbox::cluster(model_default, x)

metric_names(model_default, x, clu_default, ref)
```

```
## $internal
## [1] "silhouette"        "davies_bouldin"    "calinski_harabasz"
## 
## $external
## [1] "entropy"             "purity"              "adjusted_rand_index"
```

Run the default evaluation.

``` r
eval_default <- daltoolbox::evaluate(model_default, clu_default, ref)
eval_default$metrics
```

```
##                metric      value     goal     type
## 1          silhouette  0.5528190 maximize internal
## 2      davies_bouldin  0.6619715 minimize internal
## 3   calinski_harabasz 11.2836215 maximize internal
## 4             entropy  0.3938863 minimize external
## 5              purity  0.8933333 maximize external
## 6 adjusted_rand_index  0.7302383 maximize external
```

Now customize the same class to keep only one internal metric and one external metric. This pair is useful for teaching because `silhouette` gives a compact internal summary of cohesion and separation, while `entropy` shows how mixed each cluster is relative to a reference partition.

``` r
model_custom <- cluster_kmeans(k = 3)
model_custom$eval_internal <- list(model_custom$clu_utils$metric_silhouette)
model_custom$eval_external <- list(model_custom$clu_utils$metric_entropy)

set_example_seed()
model_custom <- daltoolbox::fit(model_custom, x)
clu_custom <- daltoolbox::cluster(model_custom, x)

metric_names(model_custom, x, clu_custom, ref)
```

```
## $internal
## [1] "silhouette"
## 
## $external
## [1] "entropy"
```

Evaluate again after the customization.

``` r
eval_custom <- daltoolbox::evaluate(model_custom, clu_custom, ref)
eval_custom$metrics
```

```
##       metric     value     goal     type
## 1 silhouette 0.5528190 maximize internal
## 2    entropy 0.3938863 minimize external
```

Available metric functions can be inspected directly from the utility object.

``` r
grep("^metric_", names(model_custom$clu_utils), value = TRUE)
```

```
##  [1] "metric_result"              "metric_from_attr"           "metric_wcss"                "metric_silhouette"          "metric_davies_bouldin"     
##  [6] "metric_calinski_harabasz"   "metric_entropy"             "metric_purity"              "metric_rand_index"          "metric_adjusted_rand_index"
## [11] "metric_noise_points"        "metric_loglik"              "metric_withinerror"         "metric_modularity"
```

What to observe
- The workflow does not change: configure, fit, cluster, and evaluate.
- What changes is only the content of `eval_internal` and `eval_external`.
- The default metric set is class-specific. The base `clusterer()` defines common defaults, and concrete classes can override them with more suitable metrics.

References
- MacQueen, J. (1967). Some Methods for classification and Analysis of Multivariate Observations.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.
- Shannon, C. E. (1948). A mathematical theory of communication.
