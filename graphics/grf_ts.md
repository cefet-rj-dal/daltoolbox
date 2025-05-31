
``` r
# installation 
install.packages("daltoobox")
```

```
## Installing package into '/home/gpca/R/x86_64-pc-linux-gnu-library/4.5'
## (as 'lib' is unspecified)
```

```
## Warning in install.packages :
##   package 'daltoobox' is not available for this version of R
## 
## A version of this package for your version of R might be available elsewhere,
## see the ideas at
## https://cran.r-project.org/doc/manuals/r-patched/R-admin.html#Installing-packages
```

``` r
# loading DAL
library(daltoolbox) 
```


``` r
library(ggplot2)
library(RColorBrewer)

# color palette
colors <- brewer.pal(4, 'Set1')

# setting the font size for all charts
font <- theme(text = element_text(size=16))
```


``` r
# Synthetic time series

x <- seq(0, 10, 0.25)
data <- data.frame(x, sin=sin(x))
head(data)
```

```
##      x       sin
## 1 0.00 0.0000000
## 2 0.25 0.2474040
## 3 0.50 0.4794255
## 4 0.75 0.6816388
## 5 1.00 0.8414710
## 6 1.25 0.9489846
```


``` r
# ts plot

# A time series plot during exploratory analysis

grf <- plot_ts(x = data$x, y = data$sin, color=colors[1])
plot(grf)
```

![plot of chunk unnamed-chunk-4](fig/grf_ts/unnamed-chunk-4-1.png)

