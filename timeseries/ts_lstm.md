## Time Series regression - Long short-term memory (LSTM)


``` r
# DAL ToolBox
# version 1.1.737



#loading DAL
library(daltoolbox)
```

### Series for studying


``` r
data(sin_data)
ts <- ts_data(sin_data$y, 10)
ts_head(ts, 3)
```

```
##             t9        t8        t7        t6        t5        t4        t3        t2        t1        t0
## [1,] 0.0000000 0.2474040 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950 0.9839859 0.9092974 0.7780732
## [2,] 0.2474040 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950 0.9839859 0.9092974 0.7780732 0.5984721
## [3,] 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950 0.9839859 0.9092974 0.7780732 0.5984721 0.3816610
```


``` r
library(ggplot2)
plot_ts(x=sin_data$x, y=sin_data$y) + theme(text = element_text(size=16))
```

![plot of chunk unnamed-chunk-3](fig/ts_lstm/unnamed-chunk-3-1.png)

### data sampling


``` r
samp <- ts_sample(ts, test_size = 5)
io_train <- ts_projection(samp$train)
io_test <- ts_projection(samp$test)
```

### Model training


``` r
model <- ts_lstm(ts_norm_gminmax(), input_size=4, epochs=10000)
model <- fit(model, x=io_train$input, y=io_train$output)
```

### Evaluation of adjustment


``` r
adjust <- predict(model, io_train$input)
```

```
## Error in prediction$tolist: $ operator is invalid for atomic vectors
```

``` r
adjust <- as.vector(adjust)
```

```
## Error: object 'adjust' not found
```

``` r
output <- as.vector(io_train$output)
ev_adjust <- evaluate(model, output, adjust)
```

```
## Error: object 'adjust' not found
```

``` r
ev_adjust$mse
```

```
## Error: object 'ev_adjust' not found
```

### Prediction of test


``` r
steps_ahead <- 1
io_test <- ts_projection(samp$test)
prediction <- predict(model, x=io_test$input, steps_ahead=steps_ahead)
```

```
## Error in prediction$tolist: $ operator is invalid for atomic vectors
```

``` r
prediction <- as.vector(prediction)
```

```
## Error: object 'prediction' not found
```

``` r
output <- as.vector(io_test$output)
if (steps_ahead > 1)
    output <- output[1:steps_ahead]

print(sprintf("%.2f, %.2f", output, prediction))
```

```
## Error: object 'prediction' not found
```

### Evaluation of test data


``` r
ev_test <- evaluate(model, output, prediction)
```

```
## Error: object 'prediction' not found
```

``` r
print(head(ev_test$metrics))
```

```
## Error: object 'ev_test' not found
```

``` r
print(sprintf("smape: %.2f", 100*ev_test$metrics$smape))
```

```
## Error: object 'ev_test' not found
```

### Plot results


``` r
yvalues <- c(io_train$output, io_test$output)
plot_ts_pred(y=yvalues, yadj=adjust, ypre=prediction) + theme(text = element_text(size=16))
```

```
## Error: object 'adjust' not found
```

