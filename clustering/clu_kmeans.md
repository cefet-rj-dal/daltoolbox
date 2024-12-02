---
title: An R Markdown document converted from "Rmd/clustering/clu_kmeans.ipynb"
output: html_document
---

# Clustering - Kmeans


```r
# DAL ToolBox
# version 1.1.727

source("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/jupyter.R")

#loading DAL
load_library("daltoolbox")  
```


```r
#load dataset
data(iris)
```

## General function to test clustering methods


```r
# setup clustering
model <- cluster_kmeans(k=3)
```


```r
# build model
model <- fit(model, iris[,1:4])
clu <- cluster(model, iris[,1:4])
table(clu)
```

```
## clu
##  1  2  3 
## 96 33 21
```


```r
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

## Influence of normalization in clustering


```r
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


```r
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

