

``` r
# Clustering - Kmeans

Sobre o método
- `cluster_kmeans`: particiona os dados em k grupos minimizando a variância intra-cluster. Sensível à escala; normalização pode melhorar os resultados.

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

Carregando dados de exemplo (`iris`).

``` r
# carregando conjunto de dados
data(iris)
```

Configuração do K-means com k=3 (uma classe por espécie em iris).

``` r
# configuração do método de clusterização
model <- cluster_kmeans(k=3)
```

Ajuste do modelo e obtenção dos rótulos de cluster.

``` r
# ajuste do modelo e rotulagem
model <- fit(model, iris[,1:4])
clu <- cluster(model, iris[,1:4])
table(clu)
```

```
## clu
##  1  2  3 
## 96 33 21
```

Avaliação externa usando os rótulos verdadeiros (`Species`).

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
## 1 1     0.999    96 0.639 
## 2 2     0        33 0     
## 3 3     0.702    21 0.0983
## 
## $clustering_entropy
## [1] 0.7375436
## 
## $data_entropy
## [1] 1.584963
```


Influência da normalização: comparar resultados após min-max.

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
## 39 50 61
```

Reavaliação com dados normalizados.

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
## 1 1     0.391    39 0.102
## 2 2     0        50 0    
## 3 3     0.777    61 0.316
## 
## $clustering_entropy
## [1] 0.4177655
## 
## $data_entropy
## [1] 1.584963
```
