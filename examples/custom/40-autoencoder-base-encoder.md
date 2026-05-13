## Autoencoder Base Encoder

This example documents the role of `autoenc_base_e` as a lightweight contract for encoder-only transformations. The base implementation is intentionally simple: it stores dimensionality information and returns the data unchanged until a specialized subclass overrides `fit()` and `transform()`.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```


``` r
x <- as.matrix(datasets::iris[, 1:4])

enc <- autoenc_base_e(input_size = 4, encoding_size = 2)
enc <- fit(enc, x)
z <- transform(enc, x)

dim(z)
```

```
## [1] 150   4
```

``` r
head(z)
```

```
##      Sepal.Length Sepal.Width Petal.Length Petal.Width
## [1,]          5.1         3.5          1.4         0.2
## [2,]          4.9         3.0          1.4         0.2
## [3,]          4.7         3.2          1.3         0.2
## [4,]          4.6         3.1          1.5         0.2
## [5,]          5.0         3.6          1.4         0.2
## [6,]          5.4         3.9          1.7         0.4
```

What to observe
- The object already participates in the DAL transformation protocol.
- The identity behavior is deliberate: this file teaches the extension contract, not a finished neural implementation.
