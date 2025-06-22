
``` r
# installation 
install.packages("daltoolbox")

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
y <- sin(x)
```


``` r
# ts plot

# A time series plot during exploratory analysis

grf <- plot_ts(x = x, y = y, color=c("red"))
plot(grf)
```

![plot of chunk unnamed-chunk-4](fig/grf_ts/unnamed-chunk-4-1.png)

