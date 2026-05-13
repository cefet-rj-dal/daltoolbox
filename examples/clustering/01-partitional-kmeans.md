About the method
- `cluster_kmeans`: partitions data into k groups by minimizing within-cluster variance. Sensitive to scale; normalization can improve results.

This is one of the best examples for learning unsupervised analysis because the method is simple but the interpretation depends heavily on preprocessing choices.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# Clustering - Kmeans

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox)  
```

Didactic goal: read the example as a study of representation and grouping, not as a prediction exercise.

Load sample data (`iris`).

``` r
# loading dataset
data(iris)
```

Configure K-means with k=3 (one cluster per species in iris).

``` r
# clustering method configuration
model <- cluster_kmeans(k=3)
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
# model fitting and labeling
set_example_seed()
model <- fit(model, iris[,1:4])
clu <- cluster(model, iris[,1:4])
table(clu)
```

```
## clu
##  1  2  3 
## 50 62 38
```

Evaluate the result using the metric configuration stored in the model. Internal metrics use only the data and the partition; external metrics compare the partition with a reference label kept aside for interpretation.

``` r
# evaluate model using internal and external metrics
eval <- evaluate(model, clu, iris$Species)
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
- The clustering is produced without using `Species`, but the labels help interpret the result afterward.
- This kind of external evaluation is pedagogical: it gives intuition about cluster quality on a known dataset.


Effect of normalization: compare results after min-max.

``` r
# Influence of normalization in clustering

iris_minmax <- transform(fit(minmax(), iris), iris)
set_example_seed()
model <- fit(model, iris_minmax[,1:4])
clu <- cluster(model, iris_minmax[,1:4])
table(clu)
```

```
## clu
##  1  2  3 
## 50 61 39
```

Re-evaluation with normalized data.

``` r
# evaluate model using internal and external metrics
eval <- evaluate(model, clu, iris_minmax$Species)
eval
```

```
## $clusters_entropy
## # A tibble: 3 × 4
##   x        ce   qtd   ceg
##   <fct> <dbl> <int> <dbl>
## 1 1     0        50 0    
## 2 2     0.777    61 0.316
## 3 3     0.391    39 0.102
## 
## $clustering_entropy
## [1] 0.4177655
## 
## $data_entropy
## [1] 1.584963
## 
## $metrics
##           metric     value     goal     type
## 1     silhouette 0.5047688 maximize internal
## 2 davies_bouldin 0.7602771 minimize internal
## 3        entropy 0.4177655 minimize external
## 4         purity 0.8866667 maximize external
```

Common mistakes
- Thinking that clustering quality depends only on the algorithm and not on the scale of the features.
- Interpreting the external labels as if they had been used during training.
- Choosing `k` mechanically without considering the analytical purpose of the grouping.

References
- MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations.
- Lloyd, S. (1982). Least squares quantization in PCM.
