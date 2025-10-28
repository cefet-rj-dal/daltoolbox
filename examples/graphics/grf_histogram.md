About the chart
- Histogram: distributes observations into bins along the x-axis; useful to visualize frequency and skewness.

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

Generate variables with distinct distributions (exponential, uniform, normal).

``` r
# Exemplos com distribuições de dados
# Usamos variáveis aleatórias para facilitar a visualização de diferentes distribuições.

# example: dataset to be plotted  
example <- data.frame(exponential = rexp(100000, rate = 1), 
                     uniform = runif(100000, min = 2.5, max = 3.5), 
                     normal = rnorm(100000, mean=5))
head(example)
```

```
##   exponential  uniform   normal
## 1   0.2185621 3.354603 3.493018
## 2   0.1654939 3.027532 6.041039
## 3   0.9174307 3.413446 4.195712
## 4   1.2676509 2.837626 4.120715
## 5   1.9322189 2.845237 3.130722
## 6   3.4969441 2.931347 5.489231
```

# Histogram

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

![plot of chunk unnamed-chunk-4](fig/grf_histogram/unnamed-chunk-4-1.png)

# Chart arrangement

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

![plot of chunk unnamed-chunk-6](fig/grf_histogram/unnamed-chunk-6-1.png)

