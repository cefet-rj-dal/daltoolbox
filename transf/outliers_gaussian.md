
``` r
# NA and Outlier analysis

Sobre a transformação
- `outliers_gaussian`: marca como outliers valores além de média ± 3 desvios padrão, assumindo distribuição aproximadamente normal.

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

```
## Error in parse(text = input): <text>:3:7: unexpected symbol
## 2: 
## 3: Sobre a
##          ^
```

Removendo outliers pela regra de 3 sigmas e inspecionando.

``` r
# Remoção de outliers usando regra Gaussiana (3σ)
# Um outlier é um valor menor que $\overline{x} - 3\,\sigma_x$ ou maior que $\overline{x} + 3\,\sigma_x$.

# A classe remove outliers em atributos numéricos.

# removendo outliers de um data frame

# código para remoção de outliers
out_obj <- outliers_gaussian() # classe para análise de outliers
out_obj <- fit(out_obj, iris)  # computa limites com base em média e desvio
iris.clean <- transform(out_obj, iris) # retorna conjunto de dados limpo

# inspection of cleaned dataset
head(iris.clean)
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
nrow(iris.clean)
```

```
## [1] 149
```

Listando observações identificadas como outliers.

``` r
# Visualizing the actual outliers

idx <- attr(iris.clean, "idx")
print(table(idx))
```

```
## idx
## FALSE  TRUE 
##   149     1
```

``` r
iris.outliers <- iris[idx,]
head(iris.outliers)
```

```
##    Sepal.Length Sepal.Width Petal.Length Petal.Width Species
## 16          5.7         4.4          1.5         0.4  setosa
```
