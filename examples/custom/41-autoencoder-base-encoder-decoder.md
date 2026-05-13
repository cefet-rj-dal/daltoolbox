## Autoencoder Base Encoder-Decoder

This example shows the role of `autoenc_base_ed` as a base contract for transformations that both compress and reconstruct the input. The default implementation is again identity-based, so the emphasis stays on the interface.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```


``` r
x <- as.matrix(datasets::iris[, 1:4])

aed <- autoenc_base_ed(input_size = 4, encoding_size = 2)
aed <- fit(aed, x)
x_rec <- transform(aed, x)

dim(x_rec)
```

```
## [1] 150   4
```

``` r
head(x_rec)
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

References
- Hinton, G. E., and Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks.
