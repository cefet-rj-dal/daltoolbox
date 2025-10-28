# Regression Random Forest

About the method
- `reg_rf`: Random Forest for regression. Averages many decision trees trained with randomness; tends to reduce variance.
- Hyperparameters: `mtry` (variables per split), `ntree` (number of trees).


``` r
# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

# Dataset for regression analysis
Load Boston dataset and inspect types/values.


``` r
library(MASS)
data(Boston)
print(t(sapply(Boston, class)))
```

```
##      crim      zn        indus     chas      nox       rm        age      
## [1,] "numeric" "numeric" "numeric" "integer" "numeric" "numeric" "numeric"
##      dis       rad       tax       ptratio   black     lstat     medv     
## [1,] "numeric" "integer" "numeric" "numeric" "numeric" "numeric" "numeric"
```

``` r
head(Boston)
```

```
##      crim zn indus chas   nox    rm  age    dis rad tax ptratio  black
## 1 0.00632 18  2.31    0 0.538 6.575 65.2 4.0900   1 296    15.3 396.90
## 2 0.02731  0  7.07    0 0.469 6.421 78.9 4.9671   2 242    17.8 396.90
## 3 0.02729  0  7.07    0 0.469 7.185 61.1 4.9671   2 242    17.8 392.83
## 4 0.03237  0  2.18    0 0.458 6.998 45.8 6.0622   3 222    18.7 394.63
## 5 0.06905  0  2.18    0 0.458 7.147 54.2 6.0622   3 222    18.7 396.90
## 6 0.02985  0  2.18    0 0.458 6.430 58.7 6.0622   3 222    18.7 394.12
##   lstat medv
## 1  4.98 24.0
## 2  9.14 21.6
## 3  4.03 34.7
## 4  2.94 33.4
## 5  5.33 36.2
## 6  5.21 28.7
```

# Optional conversion to matrix (may improve performance in some cases).

``` r
# por desempenho, você pode converter para matriz
Boston <- as.matrix(Boston)
```

# Train/test split
Random and reproducible train/test split.


``` r
# preparando amostragem aleatória
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, Boston)
boston_train <- sr$train
boston_test <- sr$test
```

# Training
Train Random Forest to predict `medv`.


``` r
model <- reg_rf("medv", mtry=7, ntree=30) # mtry: variáveis por split; ntree: nº de árvores
model <- fit(model, boston_train)
```

# Model adjustment
Training evaluation (regression metrics such as RMSE/MAE).


``` r
train_prediction <- predict(model, boston_train)
boston_train_predictand <- boston_train[,"medv"]
train_eval <- evaluate(model, boston_train_predictand, train_prediction)
print(train_eval$metrics)
```

```
##        mse      smape       R2
## 1 1.358048 0.03937262 0.984912
```

# Test
Test evaluation.


``` r
test_prediction <- predict(model, boston_test)
boston_test_predictand <- boston_test[,"medv"]
test_eval <- evaluate(model, boston_test_predictand, test_prediction)
print(test_eval$metrics)
```

```
##        mse     smape        R2
## 1 14.60407 0.1220641 0.7573084
```

