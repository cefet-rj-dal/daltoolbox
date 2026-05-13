About the transformation
- `hierarchy_cut`: converts a numeric attribute into ordered categories defined by cut points.

Didactic goal: show how a continuous variable can be turned into a simple hierarchy that is easier to interpret, summarize, or use downstream as a categorical descriptor.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
```


``` r
hc <- hierarchy_cut(
  "Sepal.Length",
  breaks = c(-Inf, 5.5, 6.5, Inf),
  labels = c("short", "medium", "long")
)

iris_h <- transform(hc, datasets::iris)
table(iris_h$Sepal.Length.Level)
```

```
## 
##  short medium   long 
##     59     61     30
```

``` r
head(iris_h[, c("Sepal.Length", "Sepal.Length.Level")])
```

```
##   Sepal.Length Sepal.Length.Level
## 1          5.1              short
## 2          4.9              short
## 3          4.7              short
## 4          4.6              short
## 5          5.0              short
## 6          5.4              short
```
