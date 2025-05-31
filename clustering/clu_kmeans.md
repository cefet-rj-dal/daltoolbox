

``` r
# Clustering - Kmeans

# installation 
install.packages("daltoobox")
```

```
## Installing package into '/home/gpca/R/x86_64-pc-linux-gnu-library/4.5'
## (as 'lib' is unspecified)
```

```
## Warning in install.packages :
##   package 'daltoobox' is not available for this version of R
## 
## A version of this package for your version of R might be available elsewhere,
## see the ideas at
## https://cran.r-project.org/doc/manuals/r-patched/R-admin.html#Installing-packages
```

``` r
# loading DAL
library(daltoolbox)  
```


``` r
# load dataset
data(iris)
```


``` r
# setup clustering
model <- cluster_kmeans(k=3)
```


``` r
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

