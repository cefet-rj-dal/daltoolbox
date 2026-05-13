About the transformation
- `sample_simple`: simple random sampling over rows or vector elements, with or without replacement.

Didactic goal: distinguish basic sampling of records from train/test partitioning. This object is useful when the task is to extract a subset, not to organize an experimental protocol.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```

Draw one sample without replacement and another with replacement.

``` r
srswor <- sample_simple(size = 10, replace = FALSE)
srswr <- sample_simple(size = 10, replace = TRUE)

sample_wor <- transform(srswor, datasets::iris$Sepal.Length)
sample_wr <- transform(srswr, datasets::iris$Sepal.Length)

sample_wor
```

```
##  [1] 5.1 6.1 7.7 5.6 4.9 5.5 5.1 4.7 6.7 5.7
```

``` r
sample_wr
```

```
##  [1] 6.1 5.0 4.4 5.1 5.0 5.5 5.1 5.7 6.2 5.6
```

References
- Cochran, W. G. (1977). Sampling Techniques.
