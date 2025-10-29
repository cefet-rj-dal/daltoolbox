About the chart
- Time series (lines): points connected by segments; highlights trend and seasonality over time/ordered axis.

Graphics environment setup.

``` r
# installation 
#install.packages("daltoolbox")

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

Synthetic series (sine and shifted cosine) for the example.

``` r
# Synthetic time series

x <- seq(0, 10, 0.25)
serie <- data.frame(x, sin=sin(x), cosine=cos(x)+5)
head(serie)
```

```
##      x       sin   cosine
## 1 0.00 0.0000000 6.000000
## 2 0.25 0.2474040 5.968912
## 3 0.50 0.4794255 5.877583
## 4 0.75 0.6816388 5.731689
## 5 1.00 0.8414710 5.540302
## 6 1.25 0.9489846 5.315322
```

Build a two-line series chart.

``` r
# Series chart

# Displays a sequence of points connected by line segments. 

# Similar to scatter, but with an x-axis ordered by time/index.

# More info: https://en.wikipedia.org/wiki/Line_chart

grf <- plot_series(serie, colors=colors[1:2]) + font
plot(grf)
```

![plot of chunk unnamed-chunk-4](fig/grf_series/unnamed-chunk-4-1.png)

References
- Wickham, H. (2016). ggplot2: Elegant Graphics for Data Analysis. Springer.
- Hyndman, R. J., and Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.
