---
title: An R Markdown document converted from "Rmd/timeseries/ts_encode-decode.ipynb"
output: html_document
---

## Time Series Encoder-Decoder


```r
# DAL ToolBox
# version 1.1.727

source("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/jupyter.R")

#loading DAL
load_library("daltoolbox")
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

![plot of chunk unnamed-chunk-4](fig/ts_encode-decode/unnamed-chunk-4-1.png)

### data sampling


```r
samp <- ts_sample(ts, test_size = 5)
train <- as.data.frame(samp$train)
test <- as.data.frame(samp$test)
```

### Model training


```r
auto <- autoenc_encode_decode(5, 3)
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
##             [,1]      [,2]      [,3]      [,4]      [,5]
## [1,] 0.009697847 0.2523975 0.4800713 0.6865125 0.8385012
## [2,] 0.248200819 0.4779633 0.6785671 0.8332504 0.9417964
## [3,] 0.474886537 0.6818241 0.8453221 0.9561558 1.0060250
## [4,] 0.676926374 0.8425300 0.9496194 1.0025870 0.9874759
## [5,] 0.843289793 0.9529023 0.9951593 0.9812074 0.9034604
## [6,] 0.949081302 0.9972236 0.9830718 0.9090387 0.7808536
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
##           [,1]      [,2]       [,3]        [,4]       [,5]
## [1,] 0.9864628 0.9171181  0.7968189  0.61868107  0.4011574
## [2,] 0.9238492 0.8032411  0.6256754  0.41917467  0.1823866
## [3,] 0.8356218 0.6338786  0.3723411  0.02444462 -0.2793030
## [4,] 0.5548474 0.3315783  0.0773633 -0.19507103 -0.4688052
## [5,] 0.3496569 0.1031587 -0.1361304 -0.33989468 -0.5531837
```

