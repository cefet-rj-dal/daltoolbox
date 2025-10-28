About the method
- `cluster_pam`: Partitioning Around Medoids. Similar to k-means but uses medoids (real points) instead of centroids, making it more robust to outliers.


``` r
# Clustering - pam

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

Load data (`iris`).

``` r
# carregando conjunto de dados
data(iris)
```

Configure PAM with k=3.

``` r
# configuração do método de clusterização
model <- cluster_pam(k=3)
```

Fit and generate cluster labels.

``` r
# ajuste do modelo e rotulagem
model <- fit(model, iris[,1:4])
clu <- cluster(model, iris[,1:4])
table(clu)
```

```
## clu
##  1  2  3 
## 50 62 38
```

External evaluation using `Species`.

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
