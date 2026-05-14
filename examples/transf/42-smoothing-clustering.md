About the technique
- `smoothing_cluster`: discretization/smoothing by class-aware clustering, so the grouping is influenced by the target class.

Discretization and smoothing
Discretization transforms continuous functions, models, variables, and equations into discrete counterparts.

Smoothing creates an approximating function to capture important patterns while reducing noise or high-frequency variation.

Defining bin intervals is an important step to enable the approximation.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```


Sample data (`iris`) to illustrate supervised clustering-based discretization/smoothing.

``` r
iris <- datasets::iris
cluster_data <- iris[, c("Sepal.Length", "Species")]
head(cluster_data)
```

```
##   Sepal.Length Species
## 1          5.1  setosa
## 2          4.9  setosa
## 3          4.7  setosa
## 4          4.6  setosa
## 5          5.0  setosa
## 6          5.4  setosa
```

Apply clustering-based smoothing and inspect bins.

``` r
# smoothing using class-aware clustering
obj <- smoothing_cluster("Species", n = 3)
set_example_seed()
obj <- fit(obj, cluster_data)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
```

```
## sl.bi
## 4.95384615384615 5.81702127659574 6.77450980392157 
##               52               47               51
```

``` r
obj$interval
```

```
## [1] 4.300000 5.469961 6.279224 7.900000
```

Evaluate conditional entropy between bins and species.

``` r
bins <- cut(iris$Sepal.Length, unique(obj$interval.adj), FALSE, include.lowest = TRUE)
entro <- evaluate(obj, bins, iris$Species)
print(entro$entropy)
```

```
## [1] 0.9083361
```

Optimizing the number of binnings

Optimize the number of bins by minimizing conditional entropy (search 1:8) and refit.

``` r
opt_obj <- smoothing_cluster("Species", n=1:8)
set_example_seed()
obj <- fit(opt_obj, cluster_data)
obj$n
```

```
## [1] 8
```


``` r
set_example_seed()
obj <- fit(obj, cluster_data)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
```

```
## sl.bi
##           4.6125            5.012 5.31818181818182  5.6047619047619             5.95            6.315 6.70869565217391 
##               16               25               11               21               22               20               23 
##            7.475 
##               12
```

References
- Han, J., Kamber, M., Pei, J. (2011). Data Mining: Concepts and Techniques. (Discretization)
