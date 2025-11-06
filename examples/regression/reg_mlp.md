About the method
- `reg_mlp`: Multilayer Perceptron (neural network) for regression.
- Hyperparameters: `size` (hidden units), `decay` (L2 regularization).

Environment setup.

``` r
# Regression MLP

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 
```

Load Boston dataset (MASS) and inspect types.

``` r
# Dataset for regression analysis

library(MASS)
data(Boston)
print(t(sapply(Boston, class)))
```

```
##      crim      zn        indus     chas      nox       rm        age       dis       rad       tax       ptratio   black     lstat    
## [1,] "numeric" "numeric" "numeric" "integer" "numeric" "numeric" "numeric" "numeric" "integer" "numeric" "numeric" "numeric" "numeric"
##      medv     
## [1,] "numeric"
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

Optional conversion to matrix.

``` r
# for performance, you can convert to matrix
Boston <- as.matrix(Boston)
```

Random and reproducible train/test split.

``` r
# preparing dataset for random sampling
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, Boston)
boston_train <- sr$train
boston_test <- sr$test
```

Train MLP: set `size` and `decay`.

``` r
# Training

model <- reg_mlp("medv", size=5, decay=0.54)
model <- fit(model, boston_train)
```

Training evaluation.

``` r
# Model adjustment

train_prediction <- predict(model, boston_train)
boston_train_predictand <- boston_train[,"medv"]
train_eval <- evaluate(model, boston_train_predictand, train_prediction)
print(train_eval$metrics)
```

```
##        mse      smape        R2
## 1 7.815809 0.09907171 0.9131656
```

Test evaluation.

``` r
# Test

test_prediction <- predict(model, boston_test)
boston_test_predictand <- boston_test[,"medv"]
test_eval <- evaluate(model, boston_test_predictand, test_prediction)
print(test_eval$metrics)
```

```
##        mse     smape        R2
## 1 17.59761 0.1396753 0.7075616
```

References
- Bishop, C. M. (1995). Neural Networks for Pattern Recognition. Oxford University Press.
