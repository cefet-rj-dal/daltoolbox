About the method
- `cluster_cmeans`: fuzzy c-means clustering, which keeps membership degrees instead of only hard assignments.

Didactic goal: keep the same clustering line of experiment and change only the notion of assignment from hard clusters to soft memberships.

Environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```

Load data and separate predictors from the reference labels used only for interpretation.

``` r
iris <- datasets::iris
x <- iris[, 1:4]
ref <- iris$Species
head(x)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width
## 1          5.1         3.5          1.4         0.2
## 2          4.9         3.0          1.4         0.2
## 3          4.7         3.2          1.3         0.2
## 4          4.6         3.1          1.5         0.2
## 5          5.0         3.6          1.4         0.2
## 6          5.4         3.9          1.7         0.4
```

Model configuration.

``` r
model <- cluster_cmeans(centers = 3, m = 2)
```

Fit the model and obtain cluster labels.

``` r
set_example_seed()
model <- fit(model, x)
clu <- cluster(model, x)
```

```
## Error in `cluster.default()`:
## ! only implemented for resamples objects
```

``` r
table(clu)
```

```
## Error:
## ! object 'clu' not found
```

Evaluate the partition.

``` r
eval <- evaluate(model, clu, ref)
```

```
## Error:
## ! object 'clu' not found
```

``` r
eval
```

```
## function (expr, envir = parent.frame(), enclos = if (is.list(envir) || 
##     is.pairlist(envir)) parent.frame() else baseenv()) 
## .Internal(eval(expr, envir, enclos))
## <bytecode: 0x5c28c1546c10>
## <environment: namespace:base>
```

Inspect the membership matrix attached to the result.

``` r
head(attr(clu, "membership"))
```

```
## Error in `h()`:
## ! error in evaluating the argument 'x' in selecting a method for function 'head': object 'clu' not found
```

References
- Bezdek, J. C. (1981). Pattern Recognition with Fuzzy Objective Function Algorithms.
