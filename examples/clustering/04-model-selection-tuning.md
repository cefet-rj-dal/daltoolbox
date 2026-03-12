About the utility
- `clu_tune`: selects hyperparameters for a clustering method. In this example, it chooses `k` for `cluster_kmeans` over a range.


``` r
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

Fit the model with a search over k=1..10 and extract the best k.

``` r
# model training with hyperparameter search
model <- clu_tune(cluster_kmeans(k = 0),  ranges = list(k = 1:10))
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
## 53 97
```

External evaluation with `Species`.

``` r
# external evaluation using ground truth labels
eval <- evaluate(model, clu, iris$Species)
eval
```

```
## $clusters_entropy
## # A tibble: 2 × 4
##   x        ce   qtd   ceg
##   <fct> <dbl> <int> <dbl>
## 1 1     0.314    53 0.111
## 2 2     0.999    97 0.646
## 
## $clustering_entropy
## [1] 0.757101
## 
## $data_entropy
## [1] 1.584963
```

References
- Satopaa, V., Albrecht, J., Irwin, D., Raghavan, B. (2011). Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior.
