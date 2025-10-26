Sobre a amostragem
- `sample_random`: separa conjuntos de treino/teste e cria folds por sorteio aleatório, mantendo apenas proporções esperadas em média.

Preparação do ambiente.

``` r
# Sampling dataset

# installation 
#install.packages("daltoolbox")

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

Dividindo em treino/teste com amostragem aleatória (distribuições podem variar por sorteio).

``` r
# Dividing a dataset with training and test

# using random sampling
tt <- train_test(sample_random(), iris)

# distribution of train
print(table(tt$train$Species))
```

```
## 
##     setosa versicolor  virginica 
##         39         40         41
```

``` r
# distribution of test
print(table(tt$test$Species))
```

```
## 
##     setosa versicolor  virginica 
##         11         10          9
```

Criando folds k-fold aleatórios e verificando distribuição por fold.

``` r
# Dividing a dataset into folds

# preparing dataset into four folds
sample <- sample_random()
folds <- k_fold(sample, iris, 4)

# distribution of folds
tbl <- NULL
for (f in folds) {
tbl <- rbind(tbl, table(f$Species))
}
print(tbl)
```

```
##      setosa versicolor virginica
## [1,]     15         14         8
## [2,]     11         15        11
## [3,]     14         10        14
## [4,]     10         11        17
```
