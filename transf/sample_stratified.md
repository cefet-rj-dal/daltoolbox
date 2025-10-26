Sobre a amostragem
- `sample_stratified`: separa treino/teste e folds preservando a proporção da variável alvo (estratificação) por categoria.

Preparação do ambiente.

``` r
# Stratified sampling dataset

# installation 
install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

Carregando dataset e visualizando distribuição de classes.

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

``` r
table(iris$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         50         50
```

Divisão estratificada em treino/teste preservando proporções de `Species`.

``` r
# Dividindo em treino e teste

# usando amostragem estratificada
tt <- train_test(sample_stratified("Species"), iris)

# distribuição do treino
print(table(tt$train$Species))
```

```
## 
##     setosa versicolor  virginica 
##         40         40         40
```

``` r
# distribuição do teste
print(table(tt$test$Species))
```

```
## 
##     setosa versicolor  virginica 
##         10         10         10
```

Criação de folds estratificados e verificação de distribuição.

``` r
# Dividindo o conjunto em folds

# usando amostragem estratificada
# preparando o conjunto em quatro folds
sample <- sample_stratified("Species")
folds <- k_fold(sample, iris, 4)

# distribuição dos folds
tbl <- NULL
for (f in folds) {
tbl <- rbind(tbl, table(f$Species))
}
print(tbl)
```

```
##      setosa versicolor virginica
## [1,]     13         13        13
## [2,]     13         13        13
## [3,]     12         12        12
## [4,]     12         12        12
```
