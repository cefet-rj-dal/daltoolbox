About the transformation
- `smoothing`: base family behind the concrete smoothing strategies.

Didactic goal: expose the role of the base object. The concrete classes such as `smoothing_inter`, `smoothing_freq`, `smoothing_quantization`, and `smoothing_cluster` inherit from this family, but the base constructor is where the generic tuning logic lives.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```

Use the base constructor with multiple candidate values for `n`, then attach the equal-interval strategy to let the built-in tuning helper choose a smoothing level.

``` r
obj <- smoothing(n = c(2, 3, 4, 5))
class(obj) <- append("smoothing_inter", class(obj))

set_example_seed()
obj <- fit(obj, datasets::iris$Sepal.Length)
smooth_values <- transform(obj, datasets::iris$Sepal.Length)

head(smooth_values)
```

```
##        1        1        1        1        1        1 
## 5.018644 5.018644 5.018644 5.018644 5.018644 5.018644
```

``` r
obj$n
```

```
## [1] 3
```

``` r
obj$interval
```

```
## [1] 4.3 5.5 6.7 7.9
```
