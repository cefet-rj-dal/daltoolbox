About the method
- `cluster_hclust`: agglomerative hierarchical clustering.

Didactic goal: keep the same clustering line of experiment and add a hierarchy that can be inspected before or after cutting the tree into clusters.

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

Model configuration.

``` r
model <- cluster_hclust(k = 3, method = "ward.D2")
```

Fit the model and obtain cluster labels.

``` r
model <- fit(model, x)
clu <- cluster(model, x)
table(clu)
```

```
## clu
##  1  2  3 
## 49 30 71
```

Evaluate the partition.

``` r
eval <- evaluate(model, clu, ref)
eval
```

```
## $clusters_entropy
## # A tibble: 3 × 4
##   x        ce   qtd   ceg
##   <fct> <dbl> <int> <dbl>
## 1 1     0        49 0    
## 2 2     0.561    30 0.112
## 3 3     0.909    71 0.430
## 
## $clustering_entropy
## [1] 0.5422445
## 
## $data_entropy
## [1] 1.584963
```

Inspect the hierarchy graphically.

``` r
grf <- plot_dendrogram(model$hc, title = "Hierarchical clustering of iris")
plot(grf)
```

![plot of chunk unnamed-chunk-6](fig/13-hierarchical-hclust/unnamed-chunk-6-1.png)

References
- Johnson, S. C. (1967). Hierarchical Clustering Schemes.
