## Tutorial 22 - Clustering Workflow

Clustering changes the analytical setting because there is no explicit target to predict during training. Even so, the workflow is still structured: choose a method, fit it, obtain cluster assignments, and inspect the result.

This tutorial also reinforces a recurring lesson from data mining: preprocessing can change unsupervised results substantially.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```

Use only the numeric attributes of `iris` for clustering.

``` r
iris <- datasets::iris
x <- iris[, 1:4]
```

Run K-means on the original scale.

``` r
model <- cluster_kmeans(k = 3)
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

``` r
evaluate(model, clu, iris$Species)
```

```
## Error:
## ! object 'clu' not found
```

Now normalize the data and repeat the same clustering procedure. Because the method is unchanged, any difference is due mainly to the representation of the data.

``` r
set_example_seed()
norm <- fit(minmax(), iris)
iris_norm <- transform(norm, iris)
x_norm <- iris_norm[, 1:4]

model_norm <- cluster_kmeans(k = 3)
set_example_seed()
model_norm <- fit(model_norm, x_norm)
clu_norm <- cluster(model_norm, x_norm)
```

```
## Error in `cluster.default()`:
## ! only implemented for resamples objects
```

``` r
table(clu_norm)
```

```
## Error:
## ! object 'clu_norm' not found
```

``` r
evaluate(model_norm, clu_norm, iris$Species)
```

```
## Error:
## ! object 'clu_norm' not found
```

This is an important lesson for beginners: in unsupervised learning, the data representation can matter as much as the algorithm.

The evaluation used above is the default evaluation of `cluster_kmeans()`. These metric lists can be customized, but that is optional and is better treated as a separate modeling choice.
