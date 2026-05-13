About the utility
- `clu_tune`: selects hyperparameters for a clustering method.
- In this example the tuned base model is `cluster_kmeans`.

Didactic goal: preserve the same clustering line of experiment and treat tuning as a change in configuration rather than as a different workflow.

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

Model configuration and tuning setup.

``` r
base_model <- cluster_kmeans(k = 2)
base_model$metric <- base_model$clu_utils$metric_silhouette
base_model$selector <- base_model$clu_utils$selector_best
base_model$eval_internal <- list(base_model$clu_utils$metric_silhouette)
base_model$eval_external <- list(base_model$clu_utils$metric_entropy)

model <- clu_tune(base_model, ranges = list(k = 2:10))
```

Fit the tuned model and inspect the selected configuration.

``` r
set_example_seed()
model <- fit(model, x)
model$k
```

```
## [1] 2
```

Generate cluster labels with the selected configuration.

``` r
clu <- cluster(model, x)
table(clu)
```

```
## clu
##  1  2 
## 97 53
```

Evaluate the tuned partition.

``` r
eval <- evaluate(model, clu, ref)
eval
```

```
## $clusters_entropy
## # A tibble: 2 × 4
##   x        ce   qtd   ceg
##   <fct> <dbl> <int> <dbl>
## 1 1     0.999    97 0.646
## 2 2     0.314    53 0.111
## 
## $clustering_entropy
## [1] 0.757101
## 
## $data_entropy
## [1] 1.584963
```

What to observe
- The workflow is still configure, fit, cluster, and evaluate.
- Tuning changes how the clustering configuration is chosen, not how the experiment is read.

References
- Satopaa, V., Albrecht, J., Irwin, D., and Raghavan, B. (2011). Finding a "Kneedle" in a Haystack.
