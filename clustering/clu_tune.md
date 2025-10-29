About the utility
- `clu_tune`: selects hyperparameters for a clustering method. In this example, it chooses `k` for `cluster_kmeans` over a range.


``` r
# Clustering - Tune Kmeans

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox)  
```

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
## [1] 7
```

Generate cluster labels with the best k.

``` r
# run with best parameter
clu <- cluster(model, iris[,1:4])
table(clu)
```

```
## clu
##  1  2  3  4  5  6  7 
## 30 16 11 22 50 17  4
```

External evaluation with `Species`.

``` r
# external evaluation using ground truth labels
eval <- evaluate(model, clu, iris$Species)
eval
```

```
## $clusters_entropy
## # A tibble: 7 × 4
##   x        ce   qtd    ceg
##   <fct> <dbl> <int>  <dbl>
## 1 1     0.469    30 0.0938
## 2 2     0        16 0     
## 3 3     0        11 0     
## 4 4     0        22 0     
## 5 5     0        50 0     
## 6 6     0.323    17 0.0366
## 7 7     0         4 0     
## 
## $clustering_entropy
## [1] 0.1303782
## 
## $data_entropy
## [1] 1.584963
```

References
- Satopaa, V., Albrecht, J., Irwin, D., Raghavan, B. (2011). Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior.
