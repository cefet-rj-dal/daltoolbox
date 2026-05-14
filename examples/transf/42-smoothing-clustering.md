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
obj <- smoothing_cluster("Species", n = 2)
set_example_seed()
obj <- fit(obj, cluster_data)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
```

```
## sl.bi
## 5.19875    6.58 
##      80      70
```

``` r
obj$interval
```

```
## [1] 4.300000 5.889375 7.900000
```

Evaluate conditional entropy between bins and species.

``` r
bins <- cut(iris$Sepal.Length, unique(obj$interval.adj), FALSE, include.lowest = TRUE)
entro <- evaluate(obj, bins, iris$Species)
print(entro$entropy)
```

```
## [1] 1.097573
```

Optimizing the number of binnings

Optimize the number of bins by minimizing conditional entropy (search 1:20) and refit.

``` r
opt_obj <- smoothing_cluster("Species", n=1:20)
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
##              4.4 4.70909090909091 5.04666666666667 5.60882352941176             6.02            6.352 6.78421052631579 
##                5               11               30               34               15               25               19 
## 7.50909090909091 
##               11
```

References
- Han, J., Kamber, M., Pei, J. (2011). Data Mining: Concepts and Techniques. (Discretization)
