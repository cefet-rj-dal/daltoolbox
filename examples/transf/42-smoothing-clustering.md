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
## 5.07230769230769 6.43294117647059 
##               65               85
```

``` r
obj$interval
```

```
## [1] 4.300000 5.682824 7.900000
```

Evaluate conditional entropy between bins and species.

``` r
bins <- cut(iris$Sepal.Length, unique(obj$interval.adj), FALSE, include.lowest = TRUE)
entro <- evaluate(obj, bins, iris$Species)
print(entro$entropy)
```

```
## [1] 1.089198
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
## [1] 19
```


``` r
set_example_seed()
obj <- fit(obj, cluster_data)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
```

```
## sl.bi
## 4.48888888888889 4.83076923076923                5 5.13076923076923 5.38571428571429              5.5 5.65714285714286 
##                9               13               10               13                7                7               14 
##             5.83                6              6.1              6.2              6.3              6.4              6.5 
##               10                6                6                4                9                7                5 
##              6.6 6.72727272727273             6.92              7.2 7.67142857142857 
##                2               11                5                5                7
```

References
- Han, J., Kamber, M., Pei, J. (2011). Data Mining: Concepts and Techniques. (Discretization)
