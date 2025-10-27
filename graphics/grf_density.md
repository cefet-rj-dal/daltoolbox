About the chart
- Density (kernel density): smoothed version of the histogram for continuous variables; highlights distribution shapes.

Graphics environment setup.

``` r
# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```


``` r
library(RColorBrewer)
# color palette
colors <- brewer.pal(4, 'Set1')

library(ggplot2)
# setting the font size for all charts
font <- theme(text = element_text(size=16))
```

Examples with distinct distributions
The following use random variables to visualize different distributions.

Generate example variables with different distributions.

``` r
# example4: dataset to be plotted  
example <- data.frame(exponential = rexp(100000, rate = 1), 
                     uniform = runif(100000, min = 2.5, max = 3.5), 
                     normal = rnorm(100000, mean=5))
head(example)
```

```
##   exponential  uniform   normal
## 1   1.4552261 2.843158 5.298216
## 2   0.1788657 2.860798 5.059152
## 3   0.4232511 3.150330 4.329689
## 4   0.6887467 2.593078 5.647122
## 5   0.7012283 3.460120 4.891822
## 6   2.2377570 2.722793 5.408614
```

# Density plot

Draws a kernel density estimate, a smoothed alternative to the histogram for continuous data.

More info: ?geom_density (R documentation)

Build densities and arrange individual charts in a grid.

``` r
options(repr.plot.width=8, repr.plot.height=5)
grf <- plot_density(example, colors=colors[1:3]) + font
```

```
## Using  as id variables
```

``` r
plot(grf)
```

![plot of chunk unnamed-chunk-4](fig/grf_density/unnamed-chunk-4-1.png)

# Chart arrangement

The `grid.arrange` function can arrange multiple previously created charts.


``` r
library(dplyr)
grfe <- plot_density(example |> dplyr::select(exponential), 
                     label_x = "exponential", color=colors[1]) + font  
```

```
## Using  as id variables
```

``` r
grfu <- plot_density(example |> dplyr::select(uniform), 
                     label_x = "uniform", color=colors[2]) + font  
```

```
## Using  as id variables
```

``` r
grfn <- plot_density(example |> dplyr::select(normal), 
                     label_x = "normal", color=colors[3]) + font 
```

```
## Using  as id variables
```


``` r
library(gridExtra)  
```

```
## Warning: package 'gridExtra' was built under R version 4.5.1
```

```
## 
## Attaching package: 'gridExtra'
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```

``` r
options(repr.plot.width=15, repr.plot.height=4)
grid.arrange(grfe, grfu, grfn, ncol=3)
```

![plot of chunk unnamed-chunk-6](fig/grf_density/unnamed-chunk-6-1.png)

