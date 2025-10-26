Sobre o gráfico
- Série temporal simples: visualização exploratória de um vetor temporal com eixo x ordenado e valores y.

Preparação do ambiente gráfico.

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

Série sintética (seno) para exemplo e plot com `plot_ts`.

``` r
# Série temporal sintética

x <- seq(0, 10, 0.25)
y <- sin(x)
```


``` r
# Gráfico de série temporal

# Visualização exploratória básica de uma série temporal

grf <- plot_ts(x = x, y = y, color=c("red"))
plot(grf)
```

![plot of chunk unnamed-chunk-4](fig/grf_ts/unnamed-chunk-4-1.png)
