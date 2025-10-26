Sobre a transformação
- Remoção de NAs: uso de `na.omit` para descartar instâncias com valores ausentes. Útil para limpeza inicial quando imputação não é desejada.

Preparação do ambiente.

``` r
# NA and Outlier analysis

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

Demonstração: inserindo um NA artificialmente e removendo linhas com NA.

``` r
# NA removal

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

``` r
nrow(iris)
```

```
## [1] 150
```


``` r
# introducing a NA to remove

iris.na <- iris
iris.na$Sepal.Length[2] <- NA
head(iris.na)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
## 1          5.1         3.5          1.4         0.2  setosa
## 2           NA         3.0          1.4         0.2  setosa
## 3          4.7         3.2          1.3         0.2  setosa
## 4          4.6         3.1          1.5         0.2  setosa
## 5          5.0         3.6          1.4         0.2  setosa
## 6          5.4         3.9          1.7         0.4  setosa
```

``` r
nrow(iris.na)
```

```
## [1] 150
```


``` r
# removing NA tuples

iris.na.omit <- na.omit(iris.na)
head(iris.na.omit)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
## 1          5.1         3.5          1.4         0.2  setosa
## 3          4.7         3.2          1.3         0.2  setosa
## 4          4.6         3.1          1.5         0.2  setosa
## 5          5.0         3.6          1.4         0.2  setosa
## 6          5.4         3.9          1.7         0.4  setosa
## 7          4.6         3.4          1.4         0.3  setosa
```

``` r
nrow(iris.na.omit)
```

```
## [1] 149
```
