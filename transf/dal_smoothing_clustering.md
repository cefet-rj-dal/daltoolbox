
``` r
# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

About the technique
- `smoothing_cluster`: discretization/smoothing by defining bins via clustering instead of fixed intervals.

# Discretization and smoothing
Discretization transforms continuous functions, models, variables, and equations into discrete counterparts.

Smoothing creates an approximating function to capture important patterns while reducing noise or high-frequency variation.

Defining bin intervals is an important step to enable the approximation.

# General function to evaluate different smoothing techniques

Sample data (`iris`) to illustrate clustering-based discretization/smoothing.

``` r
iris <- datasets::iris
head(iris)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
## 1          5.1         3.5          1.4         0.2  setosa
## 2          4.9         3.0          1.4         0.2  setosa
## 3          4.7         3.2          1.3         0.2  setosa
## 4          4.6         3.1          1.5         0.2  setosa
## 5          5.0         3.6          1.4         0.2  setosa
## 6          5.4         3.9          1.7         0.4  setosa
```

Apply clustering-based smoothing and inspect bins.

``` r
# smoothing using clustering
obj <- smoothing_cluster(n = 2)  
obj <- fit(obj, iris$Sepal.Length)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
```

```
## sl.bi
## 5.22409638554217 6.61044776119403 
##               83               67
```

``` r
obj$interval
```

```
## [1] 4.300000 5.917272 7.900000
```

Evaluate conditional entropy between bins and species.

``` r
entro <- evaluate(obj, as.factor(names(sl.bi)), iris$Species)
print(entro$entropy)
```

```
## [1] 1.12088
```

# Optimizing the number of binnings

Optimize the number of bins (search 1:20) and refit.

``` r
opt_obj <- smoothing_cluster(n=1:20)
obj <- fit(opt_obj, iris$Sepal.Length)
obj$n
```

```
## [1] 8
```


``` r
obj <- fit(obj, iris$Sepal.Length)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
```

```
## sl.bi
## 4.52727272727273 4.92380952380952 5.13076923076923 5.44285714285714 5.72916666666667         6.215625            6.725 7.50909090909091 
##               11               21               13               14               24               32               24               11
```

