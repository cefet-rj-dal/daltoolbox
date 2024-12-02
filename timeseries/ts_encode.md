---
title: An R Markdown document converted from "Rmd/timeseries/ts_encode.ipynb"
output: html_document
---

## Time Series Encoder


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
##             [,1]         [,2]      [,3]
## [1,] -0.19983050  0.150406018 0.8140124
## [2,] -0.06568242  0.003790997 0.9812956
## [3,]  0.06701577 -0.140569925 1.0961822
## [4,]  0.18564734 -0.273476303 1.1516179
## [5,]  0.28962991 -0.396227866 1.1262730
## [6,]  0.35825098 -0.487025887 1.0197763
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
##             [,1]       [,2]       [,3]
## [1,]  0.36480004 -0.5580366  0.6437690
## [2,]  0.30147842 -0.5534326  0.3799081
## [3,]  0.15360245 -0.5771290 -0.2627241
## [4,] -0.06769014 -0.2537782 -0.5324174
## [5,] -0.13025703 -0.1390109 -0.7544780
```

