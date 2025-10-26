Sobre o método
- `cluster_dbscan`: método baseado em densidade. Identifica regiões densas separadas por áreas de baixa densidade; detecta ruído e clusters de formas arbitrárias.


``` r
# Clustering - dbscan

# installation 
install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

Carregando dados (`iris`).

``` r
# carregando conjunto de dados
data(iris)
```

Configuração do DBSCAN; ajuste `minPts` (e `eps` se disponível) conforme densidade.

``` r
# configuração do método de clusterização
model <- cluster_dbscan(minPts = 3)
```

Ajuste e rótulos de cluster.

``` r
# ajuste do modelo e rotulagem
model <- fit(model, iris[,1:4])
clu <- cluster(model, iris[,1:4])
table(clu)
```

```
## clu
##  0  1  2  3  4 
## 26 47 38  4 35
```

Avaliação externa usando `Species` (atenção: DBSCAN pode marcar ruído).

``` r
# evaluate model using external metric
eval <- evaluate(model, clu, iris$Species)
eval
```

```
## $clusters_entropy
## # A tibble: 5 × 4
##   x        ce   qtd    ceg
##   <fct> <dbl> <int>  <dbl>
## 1 0     1.18     26 0.205 
## 2 1     0        47 0     
## 3 2     0        38 0     
## 4 3     0         4 0     
## 5 4     0.422    35 0.0985
## 
## $clustering_entropy
## [1] 0.3037218
## 
## $data_entropy
## [1] 1.584963
```
