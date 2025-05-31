# Clustering - dbscan
# Libraries and Datasets


``` r
# DAL ToolBox
# version 1.2.707



# loading DAL
library(daltoolbox) 
```


``` r
# load dataset
data(iris)
```

General entropy of dataset

# General function to test clustering methods


``` r
# setup clustering
model <- cluster_dbscan(minPts = 3)
```


``` r
# build model
model <- fit(model, iris[,1:4])
clu <- cluster(model, iris[,1:4])
table(clu)
```

```
## clu
##  0  1  2  3  4 
## 26 47 38  4 35
```


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

