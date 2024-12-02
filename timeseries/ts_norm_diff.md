---
title: An R Markdown document converted from "Rmd/timeseries/ts_norm_diff.ipynb"
output: html_document
---

## Normalization Diff


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
```


```r
library(ggplot2)
plot_ts(x=sin_data$x, y=sin_data$y) + theme(text = element_text(size=16))
```

![plot of chunk unnamed-chunk-3](fig/ts_norm_diff/unnamed-chunk-3-1.png)

### sliding windows


```r
sw_size <- 10
ts <- ts_data(sin_data$y, sw_size)
ts_head(ts, 3)
```

```
##             t9        t8        t7        t6        t5        t4        t3        t2        t1        t0
## [1,] 0.0000000 0.2474040 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950 0.9839859 0.9092974 0.7780732
## [2,] 0.2474040 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950 0.9839859 0.9092974 0.7780732 0.5984721
## [3,] 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950 0.9839859 0.9092974 0.7780732 0.5984721 0.3816610
```

```r
summary(ts[,10])
```

```
##        t0          
##  Min.   :-0.99929  
##  1st Qu.:-0.55091  
##  Median : 0.05397  
##  Mean   : 0.02988  
##  3rd Qu.: 0.63279  
##  Max.   : 0.99460
```


```r
library(ggplot2)
plot_ts(y=ts[,10]) + theme(text = element_text(size=16))
```

![plot of chunk unnamed-chunk-5](fig/ts_norm_diff/unnamed-chunk-5-1.png)

### normalization


```r
preproc <- ts_norm_diff()
preproc <- fit(preproc, ts)
tst <- transform(preproc, ts)
ts_head(tst, 3)
```

```
##             t8        t7        t6        t5        t4        t3        t2        t1         t0
## [1,] 0.9982009 0.9672887 0.9073861 0.8222178 0.7170790 0.5985067 0.4738732 0.3509276 0.23731412
## [2,] 0.9672887 0.9073861 0.8222178 0.7170790 0.5985067 0.4738732 0.3509276 0.2373141 0.14009662
## [3,] 0.9073861 0.8222178 0.7170790 0.5985067 0.4738732 0.3509276 0.2373141 0.1400966 0.06531964
```

```r
summary(tst[,9])
```

```
##        t0         
##  Min.   :0.00000  
##  1st Qu.:0.06333  
##  Median :0.29337  
##  Mean   :0.40975  
##  3rd Qu.:0.75129  
##  Max.   :1.00000
```

```r
plot_ts(y=ts[,9]) + theme(text = element_text(size=16))
```

![plot of chunk unnamed-chunk-6](fig/ts_norm_diff/unnamed-chunk-6-1.png)
