About the method
- `reg_rf`: Random Forest for regression. Averages many decision trees trained with randomness; tends to reduce variance.
- Hyperparameters: `mtry` (variables per split), `ntree` (number of trees).


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

Dataset for regression analysis
Load Boston dataset and inspect types/values.


``` r
library(MASS)
data(Boston)
print(t(sapply(Boston, class)))
```

```
##      crim      zn        indus     chas      nox       rm        age       dis       rad       tax       ptratio   black    
## [1,] "numeric" "numeric" "numeric" "integer" "numeric" "numeric" "numeric" "numeric" "integer" "numeric" "numeric" "numeric"
##      lstat     medv     
## [1,] "numeric" "numeric"
```

``` r
head(Boston)
```

```
##      crim zn indus chas   nox    rm  age    dis rad tax ptratio  black lstat medv
## 1 0.00632 18  2.31    0 0.538 6.575 65.2 4.0900   1 296    15.3 396.90  4.98 24.0
## 2 0.02731  0  7.07    0 0.469 6.421 78.9 4.9671   2 242    17.8 396.90  9.14 21.6
## 3 0.02729  0  7.07    0 0.469 7.185 61.1 4.9671   2 242    17.8 392.83  4.03 34.7
## 4 0.03237  0  2.18    0 0.458 6.998 45.8 6.0622   3 222    18.7 394.63  2.94 33.4
## 5 0.06905  0  2.18    0 0.458 7.147 54.2 6.0622   3 222    18.7 396.90  5.33 36.2
## 6 0.02985  0  2.18    0 0.458 6.430 58.7 6.0622   3 222    18.7 394.12  5.21 28.7
```

Optional conversion to matrix (may improve performance in some cases).

``` r
# for performance, you can convert to matrix
Boston <- as.matrix(Boston)
```

Train/test split
Random and reproducible train/test split.


``` r
# preparing random sampling
set_example_seed()
sr <- sample_random()
sr <- train_test(sr, Boston)
boston_train <- sr$train
boston_test <- sr$test
```

Training
Train Random Forest to predict `medv`.


``` r
model <- reg_rf("medv", mtry=7, ntree=30) # mtry: variables per split; ntree: number of trees
set_example_seed()
model <- fit(model, boston_train)
```

Model adjustment
Training evaluation (regression metrics such as RMSE/MAE).


``` r
train_prediction <- predict(model, boston_train)
boston_train_predictand <- boston_train[,"medv"]
train_eval <- evaluate(model, boston_train_predictand, train_prediction)
print(train_eval$metrics)
```

```
##        mse      smape        R2
## 1 1.603413 0.04054869 0.9798495
```

Test
Test evaluation.


``` r
test_prediction <- predict(model, boston_test)
boston_test_predictand <- boston_test[,"medv"]
test_eval <- evaluate(model, boston_test_predictand, test_prediction)
print(test_eval$metrics)
```

```
##        mse     smape        R2
## 1 11.15578 0.1062616 0.8913529
```

References
- Breiman, L. (2001). Random Forests. Machine Learning 45(1):5–32.
- Liaw, A. and Wiener, M. (2002). Classification and Regression by randomForest. R News.
