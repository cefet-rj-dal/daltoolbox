Sobre o gráfico
- Dispersão (scatter): avalia relação entre duas variáveis numéricas, com possibilidade de colorir por grupo/categoria.

Preparação do ambiente gráfico e paleta de cores.

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

Dados de exemplo (iris) para o gráfico.

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

Construindo gráfico de dispersão: selecione e renomeie colunas para `x`, `value` (y) e `variable` (cor).

``` r
# Gráfico de dispersão

# Usado para visualizar a relação entre duas variáveis numéricas.
# A primeira coluna do conjunto é tratada como variável no eixo X (independente) e a segunda no eixo Y (dependente);
# pode-se usar uma terceira variável categórica para colorir os pontos.

# O vetor de cores deve ter o mesmo tamanho do número de níveis/grupos.

# Mais informações: https://en.wikipedia.org/wiki/Scatter_plot

library(dplyr)

grf <- plot_scatter(
  iris |> dplyr::select(x = Sepal.Length, value = Sepal.Width, variable = Species),
  label_x = "Sepal.Length",  # rótulo do eixo X
  label_y = "Sepal.Width",   # rótulo do eixo Y
  colors=colors[1:3]          # um color para cada nível de Species
) + font
plot(grf)
```

![plot of chunk unnamed-chunk-4](fig/grf_scatter/unnamed-chunk-4-1.png)
