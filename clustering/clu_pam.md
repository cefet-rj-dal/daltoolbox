
``` r
# Clustering - pam

Sobre o método
- `cluster_pam`: Partitioning Around Medoids. Similar ao k-means, mas usa medoides (pontos reais) em vez de centróides, tornando-o mais robusto a outliers.

# installation 
install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

```
## Error in parse(text = input): <text>:3:7: unexpected symbol
## 2: 
## 3: Sobre o
##          ^
```

Carregando dados (`iris`).

``` r
# load dataset
data(iris)
```

Configuração do PAM com k=3.

``` r
# setup clustering
model <- cluster_pam(k=3)
```

Ajuste e rotulagem dos clusters.

``` r
# build model
model <- fit(model, iris[,1:4])
clu <- cluster(model, iris[,1:4])
table(clu)
```

```
## clu
##  1  2  3 
## 50 62 38
```

Avaliação externa usando `Species`.

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
