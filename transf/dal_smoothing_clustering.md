
``` r
# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

Sobre a técnica
- `smoothing_cluster`: discretização/suavização definindo bins por agrupamento (clusters) em vez de intervalos fixos.

# Discretização e suavização
Discretização é o processo de transformar funções, modelos, variáveis e equações contínuas em contrapartes discretas. 

Suavização é uma técnica que cria uma função aproximadora para capturar padrões importantes nos dados, reduzindo ruídos ou variações de alta frequência.

Uma parte importante da discretização/suavização é definir os intervalos (bins) para viabilizar a aproximação.

# Função geral para avaliar diferentes técnicas de suavização

Dados de exemplo (`iris`) para ilustrar discretização/suavização por clusters.

``` r
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

Aplicando suavização por clusterização e inspecionando bins.

``` r
# smoothing using clustering
obj <- smoothing_cluster(n = 2)  
obj <- fit(obj, iris$Sepal.Length)
sl.bi <- transform(obj, iris$Sepal.Length)
print(table(sl.bi))
```

```
## sl.bi
## 5.22409638554217 6.61044776119403 
##               83               67
```

``` r
obj$interval
```

```
## [1] 4.300000 5.917272 7.900000
```

Avaliando entropia condicional entre bins e espécie.

``` r
entro <- evaluate(obj, as.factor(names(sl.bi)), iris$Species)
print(entro$entropy)
```

```
## [1] 1.12088
```

# Optimizing the number of binnings

Otimizando o número de bins (busca em 1:20) e reaplicando o ajuste.

``` r
opt_obj <- smoothing_cluster(n=1:20)
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
## 4.52727272727273 4.92380952380952 5.13076923076923 5.44285714285714 5.72916666666667         6.215625            6.725 
##               11               21               13               14               24               32               24 
## 7.50909090909091 
##               11
```

