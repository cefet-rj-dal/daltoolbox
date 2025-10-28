About the method
- `cla_rf`: Random Forest for classification. Ensemble of decision trees trained with randomness; robust and handles heterogeneous features well.
- Hyperparameters: `mtry` (variables per split), `ntree` (number of trees).

Environment setup.

``` r
# Classification using  Random Forest

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

Load data and inspect.

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

Target `Species` levels.

``` r
# extracting the levels for the dataset
slevels <- levels(iris$Species)
slevels
```

```
## [1] "setosa"     "versicolor" "virginica"
```

# Building train and test samples via random sampling
Random train/test split.

``` r
# Construindo amostras (treino e teste) por amostragem aleatória
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test
```

Class distribution by split.

``` r
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

# Model training
Train Random Forest: tune `mtry` and `ntree`.

``` r
# Treinamento do modelo
model <- cla_rf("Species", slevels, mtry=3, ntree=5) # mtry: variáveis por split; ntree: nº de árvores
model <- fit(model, iris_train)
```

# Training evaluation

``` r
# Verificando ajuste no treino
train_prediction <- predict(model, iris_train)

# Avaliação do modelo (treino)
iris_train_predictand <- adjust_class_label(iris_train[,"Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
print(train_eval$metrics)
```

```
##   accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1    0.975 39 81  0  0         1      1           1           1  1
```

# Test evaluation

``` r
# Teste do modelo
test_prediction <- predict(model, iris_test)

iris_test_predictand <- adjust_class_label(iris_test[,"Species"])

# Avaliação no teste
 test_eval <- evaluate(model, iris_test_predictand, test_prediction)
print(test_eval$metrics)
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9666667 11 19  0  0         1      1           1           1  1
```
