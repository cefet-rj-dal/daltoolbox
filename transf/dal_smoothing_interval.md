
``` r
# installation 
install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

Sobre a técnica
- `smoothing_inter`: discretização/suavização por intervalos regulares (larguras iguais). Útil para resumir variáveis contínuas em faixas.

Dados de exemplo e ideia geral de discretização/suavização.

``` r
# Discretização e suavização
# Discretização: transformar funções, modelos, variáveis e equações contínuas em versões discretas. 

# Suavização: criar uma função aproximadora para capturar padrões importantes, reduzindo ruídos e variações de alta frequência.

# É essencial definir os intervalos (bins) para viabilizar a aproximação/discretização.

# Função geral para avaliar diferentes técnicas de suavização

iris <- datasets::iris
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

Aplicando discretização por intervalos e inspecionando os bins.

``` r
# smoothing using regular interval
obj <- smoothing_inter(n = 2)  
obj <- fit(obj, iris$Sepal.Length)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
```

```
## sl.bi
## 5.32842105263158 6.73272727272727 
##               95               55
```

``` r
obj$interval
```

```
## [1] 4.3 6.1 7.9
```

Avaliando entropia condicional entre bins e espécie.

``` r
entro <- evaluate(obj, as.factor(names(sl.bi)), iris$Species)
print(entro$entropy)
```

```
## [1] 1.191734
```

Otimizando o número de bins (busca em 1:20) e aplicando novamente.

``` r
# Optimizing the number of binnings

opt_obj <- smoothing_inter(n=1:20)
obj <- fit(opt_obj, iris$Sepal.Length)
obj$n
```

```
## [1] 8
```


``` r
obj <- fit(obj, iris$Sepal.Length)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
```

```
## sl.bi
## 4.52727272727273 5.00294117647059             5.49 5.88333333333333            6.352 6.76666666666667 7.23333333333333 
##               11               34               20               30               25               18                6 
## 7.71666666666667 
##                6
```
