# Leveraging Experiment Lines to Data Analytics

## Time Series regression - tune - example

Our objective here is to generate a model that is able to do time series forecasting.

Configuring the environment:


``` r
# DAL ToolBox
# version 1.1.737



#loading DAL Toolbox
library(daltoolbox)

#load required library
library(ggplot2)
```

### Cosine time series for studying

Generate a cosine time series to use in the example, it starts at 0 (zero) and goes up to 25 (twenty-five).


``` r
i <- seq(0, 25, 0.25)
x <- cos(i)
```

Plots the time series:


``` r
plot_ts(x=i, y=x) + theme(text = element_text(size=16))
```

![plot of chunk unnamed-chunk-3](fig/ts_tune/unnamed-chunk-3-1.png)

### Sliding windows

Creates a matrix representing a sliding window to be used in the process of training the model. Each row of the matrix represents one moment of the sliding window, with 10 (ten) elements as attributes (t9, t8, t7, ..., t0).


``` r
sw_size <- 10
ts <- ts_data(x, sw_size)
ts_head(ts, 3)
```

```
##             t9        t8        t7        t6        t5         t4         t3         t2         t1         t0
## [1,] 1.0000000 0.9689124 0.8775826 0.7316889 0.5403023  0.3153224  0.0707372 -0.1782461 -0.4161468 -0.6281736
## [2,] 0.9689124 0.8775826 0.7316889 0.5403023 0.3153224  0.0707372 -0.1782461 -0.4161468 -0.6281736 -0.8011436
## [3,] 0.8775826 0.7316889 0.5403023 0.3153224 0.0707372 -0.1782461 -0.4161468 -0.6281736 -0.8011436 -0.9243024
```

### Data sampling

Samples data into train and test.


``` r
test_size <- 1
samp <- ts_sample(ts, test_size)
ts_head(samp$train, 3)
```

```
##             t9        t8        t7        t6        t5         t4         t3         t2         t1         t0
## [1,] 1.0000000 0.9689124 0.8775826 0.7316889 0.5403023  0.3153224  0.0707372 -0.1782461 -0.4161468 -0.6281736
## [2,] 0.9689124 0.8775826 0.7316889 0.5403023 0.3153224  0.0707372 -0.1782461 -0.4161468 -0.6281736 -0.8011436
## [3,] 0.8775826 0.7316889 0.5403023 0.3153224 0.0707372 -0.1782461 -0.4161468 -0.6281736 -0.8011436 -0.9243024
```

``` r
ts_head(samp$test)
```

```
##              t9        t8         t7          t6        t5       t4       t3        t2        t1        t0
## [1,] -0.7256268 -0.532833 -0.3069103 -0.06190529 0.1869486 0.424179 0.635036 0.8064095 0.9276444 0.9912028
```

### Model training

Tune optimizes a learner hyperparameter, no matter which one. This way, in this example, an ELM is used in the hyperparameters tuning using an appropriate range. The result of tunning is an ELM model for the training set.


``` r
# Setup for tunning using ELM
tune <- ts_tune(input_size=c(3:5), base_model = ts_elm(ts_norm_gminmax()))
ranges <- list(nhid = 1:5, actfun=c('sig', 'radbas', 'tribas', 'relu', 'purelin'))
```

In [6] is using ts_elm() as base_model, but all time series models can be used. It is simple as changing the constructor.

An LSTM could be used, as shown at In [7], as lines of comments. This example clarifies how to provide variability on workflow models by simply changing constructors.

Options of ranges for all time series models are presented in the end of this notebook.

Input size options should be between 1 and sw_size-2.


``` r
# tune <- ts_tune(input_size=c(3:5), base_model = ts_lstm(ts_norm_gminmax()))
# ranges <- list(input_size = 1:10, epochs=10000)
```

The prediction output using the training set can be used to evaluate the model's adjustment level to the training data:


``` r
io_train <- ts_projection(samp$train)

# Generic model tunning
model <- fit(tune, x=io_train$input, y=io_train$output, ranges)
```

### Evaluation of adjustment


``` r
adjust <- predict(model, io_train$input)
ev_adjust <- evaluate(model, io_train$output, adjust)
print(head(ev_adjust$metrics))
```

```
##            mse        smape R2
## 1 3.235577e-30 7.125308e-15  1
```

### Prediction of test


``` r
steps_ahead <- 1
io_test <- ts_projection(samp$test)
prediction <- predict(model, x=io_test$input, steps_ahead=steps_ahead)
prediction <- as.vector(prediction)

output <- as.vector(io_test$output)
if (steps_ahead > 1)
    output <- output[1:steps_ahead]

print(sprintf("%.2f, %.2f", output, prediction))
```

```
## [1] "0.99, 0.99"
```

### Evaluation of test data


``` r
ev_test <- evaluate(model, output, prediction)
print(head(ev_test$metrics))
```

```
##           mse        smape   R2
## 1 1.49144e-30 1.232084e-15 -Inf
```

``` r
print(sprintf("smape: %.2f", 100*ev_test$metrics$smape))
```

```
## [1] "smape: 0.00"
```

### Plot results

The plot shows results of the prediction. 


``` r
yvalues <- c(io_train$output, io_test$output)
plot_ts_pred(y=yvalues, yadj=adjust, ypre=prediction) + theme(text = element_text(size=16))
```

![plot of chunk unnamed-chunk-12](fig/ts_tune/unnamed-chunk-12-1.png)

### Otions for machine learning

Options of ranges for all time series models:


``` r
### Ranges for ELM
ranges_elm <- list(nhid = 1:20, actfun=c('sig', 'radbas', 'tribas', 'relu', 'purelin'))

### Ranges for MLP
ranges_mlp <- list(size = 1:10, decay = seq(0, 1, 1/9), maxit=10000)

### Ranges for RF
ranges_rf <- list(nodesize=1:10, ntree=1:10)

### Ranges for SVM
ranges_svm <- list(kernel=c("radial", "poly", "linear", "sigmoid"), epsilon=seq(0, 1, 0.1), cost=seq(20, 100, 20))

### Ranges for LSTM
ranges_lstm <- list(input_size = 1:10, epochs=10000)

### Ranges for CNN
ranges_cnn <- list(input_size = 1:10, epochs=10000)
```

