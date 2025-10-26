
``` r
# installation 
install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

# for ploting
library(ggplot2)
library(dplyr)
```

Sobre a técnica
- `fit_curvature_min`: calcula a curvatura pela segunda derivada de um spline suavizado sobre a sequência e retorna a posição de curvatura mínima em curvas crescentes; útil para achar um ponto de compromisso onde ganhos adicionais tornam-se marginais.

Carregando dados de exemplo (PCA no dataset wine) e montando curva de variância acumulada.

``` r
wine <- get(load(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/develop/wine.RData")))
```

```
## Warning in load(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/develop/wine.RData")): cannot open URL
## 'https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/develop/wine.RData': HTTP status was '429 Unknown Error'
```

```
## Error in load(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/develop/wine.RData")): cannot open the connection to 'https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/develop/wine.RData'
```

``` r
head(wine)
```

```
## Error: object 'wine' not found
```

# Exemplo: componentes da PCA
Variância acumulada da PCA: as primeiras dimensões concentram alta variância; adicionar muitas dimensões traz ganhos marginais. 
O objetivo é estabelecer um ponto de compromisso (trade-off).


``` r
pca_res = prcomp(wine[,2:ncol(wine)], center=TRUE, scale.=TRUE)
```

```
## Error: object 'wine' not found
```

``` r
y <- cumsum(pca_res$sdev^2/sum(pca_res$sdev^2)) # variância acumulada
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
res <- transform(myfit, y)  # retorna índice ótimo (joelho)
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
