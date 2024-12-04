## Time Series Encoder


```r
# DAL ToolBox
# version 1.1.727



#loading DAL
library(daltoolbox)("daltoolbox")
```

```
## Error in eval(expr, envir, enclos): attempt to apply non-function
```

### Series for studying


```r
data(sin_data)
sin_data$y[39] <- sin_data$y[39]*6
```


```r
sw_size <- 5
ts <- ts_data(sin_data$y, sw_size)
ts_head(ts, 3)
```

```
##             t4        t3        t2        t1        t0
## [1,] 0.0000000 0.2474040 0.4794255 0.6816388 0.8414710
## [2,] 0.2474040 0.4794255 0.6816388 0.8414710 0.9489846
## [3,] 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950
```


```r
library(ggplot2)
plot_ts(x=sin_data$x, y=sin_data$y) + theme(text = element_text(size=16))
```

![plot of chunk unnamed-chunk-4](fig/ts_encode/unnamed-chunk-4-1.png)

### data sampling


```r
samp <- ts_sample(ts, test_size = 5)
train <- as.data.frame(samp$train)
test <- as.data.frame(samp$test)
```

### Model training


```r
auto <- autoenc_encode(5, 3)
auto <- fit(auto, train)
```

### Evaluation of encoding


```r
print(head(train))
```

```
##          t4        t3        t2        t1        t0
## 1 0.0000000 0.2474040 0.4794255 0.6816388 0.8414710
## 2 0.2474040 0.4794255 0.6816388 0.8414710 0.9489846
## 3 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950
## 4 0.6816388 0.8414710 0.9489846 0.9974950 0.9839859
## 5 0.8414710 0.9489846 0.9974950 0.9839859 0.9092974
## 6 0.9489846 0.9974950 0.9839859 0.9092974 0.7780732
```

```r
result <- transform(auto, train)
print(head(result))
```

```
##           [,1]       [,2]      [,3]
## [1,] 0.0677183 -0.6762174 1.0282019
## [2,] 0.3043296 -0.9997389 1.0442926
## [3,] 0.5224816 -1.2701842 0.9764421
## [4,] 0.6991041 -1.4687785 0.8372102
## [5,] 0.8104581 -1.5820323 0.6499083
## [6,] 0.8447057 -1.6096097 0.4332939
```

### Encoding of test


```r
print(head(test))
```

```
##          t4        t3         t2         t1         t0
## 1 0.9893582 0.9226042  0.7984871  0.6247240  0.4121185
## 2 0.9226042 0.7984871  0.6247240  0.4121185  0.1738895
## 3 0.7984871 0.6247240  0.4121185  0.1738895 -0.4509067
## 4 0.6247240 0.4121185  0.1738895 -0.4509067 -0.3195192
## 5 0.4121185 0.1738895 -0.4509067 -0.3195192 -0.5440211
```

```r
result <- transform(auto, test)
print(head(result))
```

```
##            [,1]       [,2]        [,3]
## [1,]  0.7437662 -1.3976041 -0.09033697
## [2,]  0.5745521 -1.1749763 -0.36347350
## [3,]  0.3190219 -0.9235739 -0.81161129
## [4,] -0.1885568 -0.4234360 -0.89012259
## [5,] -0.3831522 -0.1132880 -1.01046491
```

