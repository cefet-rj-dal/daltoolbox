About the utility
- `clu_tune`: selects hyperparameters for a clustering method.
- In this example, it chooses `k` for `cluster_kmeans` over a range, using the metric and selector configured in the base model.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# Clustering - Tune Kmeans

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox)  
```

Didactic goal: read this example as an unsupervised workflow. The emphasis is not on predicting a known label during training, but on understanding how the method groups the data and how preprocessing affects that grouping.

Load data (`iris`).

``` r
data(iris)
```

Fit the model with a search over `k = 2..10` and extract the best k.

``` r
# model training with hyperparameter search
base_model <- cluster_kmeans(k = 2)
base_model$metric <- base_model$clu_utils$metric_silhouette
base_model$selector <- base_model$clu_utils$selector_best
base_model$eval_internal <- list(base_model$clu_utils$metric_silhouette)
base_model$eval_external <- list(base_model$clu_utils$metric_entropy)
model <- clu_tune(base_model, ranges = list(k = 2:10))
set_example_seed()
model <- fit(model, iris[,1:4])
model$k
```

```
## [1] 2
```

Generate cluster labels with the best k.

``` r
# run with best parameter
clu <- cluster(model, iris[,1:4])
table(clu)
```

```
## clu
##  1  2 
## 97 53
```

Evaluate the tuned result internally and externally.

``` r
# internal and external evaluation
eval <- evaluate(model, clu, iris$Species)
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
## 
## $metrics
##       metric     value     goal     type
## 1 silhouette 0.6810462 maximize internal
## 2    entropy 0.7571010 minimize external
```

References
- Satopaa, V., Albrecht, J., Irwin, D., Raghavan, B. (2011). Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior.
