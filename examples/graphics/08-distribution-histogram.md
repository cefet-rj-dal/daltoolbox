About the chart
- Histogram: distributes observations into bins along the x-axis; useful to visualize frequency and skewness.

Graphics environment setup.

``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
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

Generate variables with distinct distributions (exponential, uniform, normal).

``` r
# Examples with data distributions
# We use random variables to facilitate visualization of different distributions.

# example: dataset to be plotted  
example <- data.frame(exponential = rexp(100000, rate = 1), 
                     uniform = runif(100000, min = 2.5, max = 3.5), 
                     normal = rnorm(100000, mean=5))
head(example)
```

```
##   exponential  uniform   normal
## 1   0.1983368 3.259554 4.896481
## 2   0.6608953 3.171106 5.187828
## 3   0.2834910 3.423250 4.389551
## 4   0.0381919 2.531736 3.741199
## 5   0.4731766 2.619622 4.529254
## 6   1.4636271 2.730006 4.735878
```

Histogram

Visualize the distribution of a continuous variable by binning the x-axis and counting observations per bin. `geom_histogram()` displays counts as bars.
More info: ?geom_histogram (R documentation)

Build histograms and arrange multiple charts side by side.

``` r
library(dplyr)

grf <- plot_hist(example |> dplyr::select(exponential), 
                  label_x = "exponential", color=colors[1]) + font
```

```
## Using  as id variables
```

``` r
options(repr.plot.width=5, repr.plot.height=4)
plot(grf)
```

![plot of chunk unnamed-chunk-4](fig/08-distribution-histogram/unnamed-chunk-4-1.png)

Chart arrangement

Use `grid.arrange` to place the generated charts side by side.


``` r
grfe <- plot_hist(example |> dplyr::select(exponential), 
                  label_x = "exponential", color=colors[1]) + font
```

```
## Using  as id variables
```

``` r
grfu <- plot_hist(example |> dplyr::select(uniform), 
                  label_x = "uniform", color=colors[1]) + font  
```

```
## Using  as id variables
```

``` r
grfn <- plot_hist(example |> dplyr::select(normal), 
                  label_x = "normal", color=colors[1]) + font 
```

```
## Using  as id variables
```


``` r
library(gridExtra)  

options(repr.plot.width=15, repr.plot.height=4)
grid.arrange(grfe, grfu, grfn, ncol=3)
```

![plot of chunk unnamed-chunk-6](fig/08-distribution-histogram/unnamed-chunk-6-1.png)

References
- Freedman, D., and Diaconis, P. (1981). On the histogram as a density estimator: L2 theory. Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete.
- Wickham, H. (2016). ggplot2: Elegant Graphics for Data Analysis. Springer.
