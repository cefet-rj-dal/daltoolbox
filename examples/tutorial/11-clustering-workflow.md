## Tutorial 11 - Clustering Workflow

Clustering changes the analytical setting because there is no explicit target to predict during training. Even so, the workflow is still structured: choose a method, fit it, obtain cluster assignments, and inspect the result.

This tutorial also reinforces a recurring lesson from data mining: preprocessing can change unsupervised results substantially.


``` r
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
model <- fit(model, x)
clu <- cluster(model, x)

table(clu)
```

```
## clu
##  1  2  3 
## 38 50 62
```

``` r
evaluate(model, clu, iris$Species)
```

```
## $clusters_entropy
## # A tibble: 3 × 4
##   x        ce   qtd    ceg
##   <fct> <dbl> <int>  <dbl>
## 1 1     0.297    38 0.0754
## 2 2     0        50 0     
## 3 3     0.771    62 0.319 
## 
## $clustering_entropy
## [1] 0.3938863
## 
## $data_entropy
## [1] 1.584963
```

Now normalize the data and repeat the same clustering procedure. Because the method is unchanged, any difference is due mainly to the representation of the data.

``` r
norm <- fit(minmax(), iris)
iris_norm <- transform(norm, iris)
x_norm <- iris_norm[, 1:4]

model_norm <- cluster_kmeans(k = 3)
model_norm <- fit(model_norm, x_norm)
clu_norm <- cluster(model_norm, x_norm)

table(clu_norm)
```

```
## clu_norm
##  1  2  3 
## 61 50 39
```

``` r
evaluate(model_norm, clu_norm, iris$Species)
```

```
## $clusters_entropy
## # A tibble: 3 × 4
##   x        ce   qtd   ceg
##   <fct> <dbl> <int> <dbl>
## 1 1     0.777    61 0.316
## 2 2     0        50 0    
## 3 3     0.391    39 0.102
## 
## $clustering_entropy
## [1] 0.4177655
## 
## $data_entropy
## [1] 1.584963
```

This is an important lesson for beginners: in unsupervised learning, the data representation can matter as much as the algorithm.
