

``` r
# Clustering - Tune Kmeans

# installation 
install.packages("daltoolbox")
```

```

```

``` r
# loading DAL
library(daltoolbox)  
```


``` r
data(iris)
```


``` r
# fit model
model <- clu_tune(cluster_kmeans(k = 0))
ranges <- list(k = 1:10)
model <- fit(model, iris[,1:4], ranges)
model$k
```

```
## [1] 9
```


``` r
# run model
clu <- cluster(model, iris[,1:4])
table(clu)
```

```
## clu
##  1  2  3  4  5  6  7  8  9 
## 12  9 50 12 20  4 17 14 12
```


``` r
# evaluate model using external metric
eval <- evaluate(model, clu, iris$Species)
eval
```

```
## $clusters_entropy
## # A tibble: 9 × 4
##   x        ce   qtd    ceg
##   <fct> <dbl> <int>  <dbl>
## 1 1     0        12 0     
## 2 2     0         9 0     
## 3 3     0        50 0     
## 4 4     0        12 0     
## 5 5     0.286    20 0.0382
## 6 6     0         4 0     
## 7 7     0.672    17 0.0762
## 8 8     0        14 0     
## 9 9     0        12 0     
## 
## $clustering_entropy
## [1] 0.1143797
## 
## $data_entropy
## [1] 1.584963
```

