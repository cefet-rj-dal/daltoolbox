
``` r
# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

Sobre a técnica
- `smoothing_freq`: discretização/suavização por frequência (quantis), gerando bins com contagens semelhantes.

Dados de exemplo e ideia geral.

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

Aplicando discretização por frequência e inspecionando intervalos.

``` r
# smoothing using regular frequency

obj <- smoothing_freq(n = 2)  
obj <- fit(obj, iris$Sepal.Length)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
```

```
## sl.bi
## 5.19875    6.58 
##      80      70
```

``` r
obj$interval
```

```
## [1] 4.3 5.8 7.9
```

Avaliando entropia condicional.

``` r
entro <- evaluate(obj, as.factor(names(sl.bi)), iris$Species)
print(entro$entropy)
```

```
## [1] 1.097573
```

Otimizando número de bins e aplicando novamente.

``` r
# Optimizing the number of binnings

opt_obj <- smoothing_freq(n=1:20)
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
## 4.69090909090909 5.04736842105263 5.38888888888889  5.7047619047619             6.02            6.315             6.65 
##               22               19               18               21               15               20               18 
## 7.31176470588235 
##               17
```
