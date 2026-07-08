About the technique
- `smoothing_quantization`: discretization/smoothing by one-dimensional k-means quantization, without using the class label.

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

Sample data (`iris`) to illustrate one-dimensional quantization.

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

Apply k-means quantization and inspect bins.

``` r
obj <- smoothing_quantization(n = 2)
set_example_seed()
obj <- fit(obj, iris$Sepal.Length)
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

Evaluate conditional entropy between bins and species after the quantization step.

``` r
bins <- cut(iris$Sepal.Length, unique(obj$interval.adj), FALSE, include.lowest = TRUE)
entro <- evaluate(obj, bins, iris$Species)
print(entro$entropy)
```

```
## [1] 1.097573
```

Optimizing the number of binnings

Optimize the number of bins by the MSE elbow heuristic (search 1:20) and refit.

``` r
opt_obj <- smoothing_quantization(n=1:20)
set_example_seed()
obj <- fit(opt_obj, iris$Sepal.Length)
obj$n
```

```
## [1] 8
```


``` r
set_example_seed()
obj <- fit(obj, iris$Sepal.Length)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
```

```
## sl.bi
##              4.4 4.70909090909091 5.04666666666667 5.60882352941176             6.02            6.352 6.78421052631579 7.50909090909091 
##                5               11               30               34               15               25               19               11
```

References
- MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations.
