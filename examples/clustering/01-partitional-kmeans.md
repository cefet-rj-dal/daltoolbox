About the method
- `cluster_kmeans`: partitions data into k groups by minimizing within-cluster variance. Sensitive to scale; normalization can improve results.

This is one of the best examples for learning unsupervised analysis because the method is simple but the interpretation depends heavily on preprocessing choices.


``` r
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
```

Fit the model and obtain cluster labels.

``` r
# model fitting and labeling
model <- fit(model, iris[,1:4])
clu <- cluster(model, iris[,1:4])
table(clu)
```

```
## clu
##  1  2  3 
## 50 62 38
```

External evaluation using true labels (`Species`).

``` r
# evaluate model using external metric
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
```

What to observe
- The clustering is produced without using `Species`, but the labels help interpret the result afterward.
- This kind of external evaluation is pedagogical: it gives intuition about cluster quality on a known dataset.


Effect of normalization: compare results after min-max.

``` r
# Influence of normalization in clustering

iris_minmax <- transform(fit(minmax(), iris), iris)
model <- fit(model, iris_minmax[,1:4])
clu <- cluster(model, iris_minmax[,1:4])
table(clu)
```

```
## clu
##  1  2  3 
## 50 39 61
```

Re-evaluation with normalized data.

``` r
# evaluate model using external metric

eval <- evaluate(model, clu, iris_minmax$Species)
eval
```

```
## $clusters_entropy
## # A tibble: 3 × 4
##   x        ce   qtd   ceg
##   <fct> <dbl> <int> <dbl>
## 1 1     0        50 0    
## 2 2     0.391    39 0.102
## 3 3     0.777    61 0.316
## 
## $clustering_entropy
## [1] 0.4177655
## 
## $data_entropy
## [1] 1.584963
```

Common mistakes
- Thinking that clustering quality depends only on the algorithm and not on the scale of the features.
- Interpreting the external labels as if they had been used during training.
- Choosing `k` mechanically without considering the analytical purpose of the grouping.

References
- MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations.
- Lloyd, S. (1982). Least squares quantization in PCM.
