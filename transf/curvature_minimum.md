
``` r
# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

# for ploting
library(ggplot2)
library(dplyr)
```


``` r
wine <- get(load(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/develop/data/wine.RData")))
```

```
## Warning in load(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/develop/data/wine.RData")): cannot open URL
## 'https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/develop/data/wine.RData': HTTP status was '404 Not Found'
```

```
## Error in load(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/develop/data/wine.RData")): cannot open the connection to 'https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/develop/data/wine.RData'
```

``` r
head(wine)
```

```
## Error: object 'wine' not found
```

# Example: PCA components
Cummulative variance of PCA: First dimensions have high variance. However, adding more dimensions does not bring much benefit in terms of cummulative variance. 
The goal is to establish a trade-off.


``` r
pca_res = prcomp(wine[,2:ncol(wine)], center=TRUE, scale.=TRUE)
```

```
## Error: object 'wine' not found
```

``` r
y <- cumsum(pca_res$sdev^2/sum(pca_res$sdev^2))
```

```
## Error: object 'pca_res' not found
```

``` r
x <- 1:length(y)
```

```
## Error: object 'y' not found
```


``` r
dat <- data.frame(x, value = y, variable = "PCA")
```

```
## Error: object 'x' not found
```

``` r
dat$variable <- as.factor(dat$variable)
```

```
## Error: object 'dat' not found
```

``` r
head(dat)
```

```
## Error: object 'dat' not found
```


``` r
grf <- plot_scatter(dat, label_x = "dimensions", label_y = "cumulative variance", colors="black") + 
    theme(text = element_text(size=16))
```

```
## Error: object 'dat' not found
```

``` r
plot(grf)
```

```
## Error: object 'grf' not found
```

# Minimum curvature
If the curve is increasing, use minimum curvature analysis. 
It brings a trade-off between having lower x values (with not so high y values) and having higher x values (not having to much increase in y values). 


``` r
myfit <- fit_curvature_min()
res <- transform(myfit, y)
```

```
## Error: object 'y' not found
```

``` r
head(res)
```

```
## Error: object 'res' not found
```


``` r
plot(grf + geom_vline(xintercept = res$x, linetype="dashed", color = "red", size=0.5))
```

```
## Error: object 'grf' not found
```

