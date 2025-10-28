About the utility
- `cla_tune`: performs hyperparameter search for a classifier over ranges defined in `ranges`.
- The example below tunes a `cla_svm` varying `epsilon`, `cost`, and `kernel`.

Environment setup.

``` r
# Tuning de Classificação 

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

Sample data and target levels.

``` r
# Conjunto de dados para classificação

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
# extracting the levels for the dataset
slevels <- levels(iris$Species)
slevels
```

```
## [1] "setosa"     "versicolor" "virginica"
```

# Train/test split
Random split for tuning validation.

``` r
# preparando amostragem aleatória
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test

tbl <- rbind(table(iris[,"Species"]),
             table(iris_train[,"Species"]),
             table(iris_test[,"Species"]))
rownames(tbl) <- c("dataset", "training", "test")
head(tbl)
```

```
##          setosa versicolor virginica
## dataset      50         50        50
## training     39         38        43
## test         11         12         7
```

# Hyperparameter grid and search training

``` r
# Treinamento com busca de hiperparâmetros
tune <- cla_tune(cla_svm("Species", slevels), 
  ranges = list(epsilon=seq(0,1,0.2), cost=seq(20,100,20), kernel = c("linear", "radial", "polynomial", "sigmoid")))

model <- fit(tune, iris_train)
```

# Training evaluation with the best configuration

``` r
# Avaliação no treino
train_prediction <- predict(model, iris_train)

iris_train_predictand <- adjust_class_label(iris_train[,"Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
print(train_eval$metrics)
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9833333 39 81  0  0         1      1           1           1  1
```

# Test evaluation

``` r
# Avaliação no teste
test_prediction <- predict(model, iris_test)

iris_test_predictand <- adjust_class_label(iris_test[,"Species"])

# Evaluating # setosa as primary class
test_eval <- evaluate(model, iris_test_predictand, test_prediction)
print(test_eval$metrics)
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9333333 11 19  0  0         1      1           1           1  1
```

# Example grids for other models

``` r
# Opções de grids para outros modelos
# knn
ranges <- list(k=1:20)

# mlp
ranges <- list(size=1:10, decay=seq(0, 1, 0.1))

# rf
ranges <- list(mtry=1:3, ntree=1:10)
```
