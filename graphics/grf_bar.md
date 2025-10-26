Sobre o gráfico
- Barras (bar): compara valores agregados por categorias. Útil para médias, contagens e totais por grupo.

Preparação do ambiente gráfico e paleta de cores.

``` r
# installation 
install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```


``` r
library(RColorBrewer)
library(ggplot2)

colors <- brewer.pal(4, 'Set1')

# setting the font size for all charts
font <- theme(text = element_text(size=16))
```

Dados de exemplo agregados por `Species`.

``` r
# conjunto de dados iris para o exemplo
head(iris)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
## 1          5.1         3.5          1.4         0.2  setosa
## 2          4.9         3.0          1.4         0.2  setosa
## 3          4.7         3.2          1.3         0.2  setosa
## 4          4.6         3.1          1.5         0.2  setosa
## 5          5.0         3.6          1.4         0.2  setosa
## 6          5.4         3.9          1.7         0.4  setosa
```


``` r
library(dplyr)
data <- iris |> group_by(Species) |> summarize(Sepal.Length=mean(Sepal.Length))
head(data)
```

```
## # A tibble: 3 × 2
##   Species    Sepal.Length
##   <fct>             <dbl>
## 1 setosa             5.01
## 2 versicolor         5.94
## 3 virginica          6.59
```

Gráfico de barras básico e variação com barras verticais.

``` r
# Gráfico de barras

# Apresenta dados categóricos com barras proporcionais ao valor agregado (contagem, média, etc.).

# Mais informações: https://en.wikipedia.org/wiki/Bar_chart

grf <- plot_bar(data, colors=colors[1]) + font
plot(grf)
```

![plot of chunk unnamed-chunk-5](fig/grf_bar/unnamed-chunk-5-1.png)


``` r
# As barras podem ser invertidas (horizontais/verticais) com coord_flip().
grf <- grf + coord_flip()
plot(grf)
```

![plot of chunk unnamed-chunk-6](fig/grf_bar/unnamed-chunk-6-1.png)

Colorindo cada barra por espécie.

``` r
# Bar graph with one color for each species
grf <- plot_bar(data, colors=colors[1:3]) + font
plot(grf)
```

![plot of chunk unnamed-chunk-7](fig/grf_bar/unnamed-chunk-7-1.png)
