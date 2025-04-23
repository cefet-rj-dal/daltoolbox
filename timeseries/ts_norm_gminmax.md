## Normalization Global Min-Max


``` r
# DAL ToolBox
# version 1.1.737



#loading DAL
library(daltoolbox) 
```

### Series for studying


``` r
data(sin_data)
```


``` r
library(ggplot2)
plot_ts(x=sin_data$x, y=sin_data$y) + theme(text = element_text(size=16))
```

![plot of chunk unnamed-chunk-3](fig/ts_norm_gminmax/unnamed-chunk-3-1.png)

### sliding windows


``` r
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

``` r
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


``` r
library(ggplot2)
plot_ts(y=ts[,10]) + theme(text = element_text(size=16))
```

![plot of chunk unnamed-chunk-5](fig/ts_norm_gminmax/unnamed-chunk-5-1.png)

### normalization


``` r
preproc <- ts_norm_gminmax()
preproc <- fit(preproc, ts)
tst <- transform(preproc, ts)
ts_head(tst, 3)
```

```
##             t9        t8        t7        t6        t5        t4        t3        t2        t1        t0
## [1,] 0.5004502 0.6243512 0.7405486 0.8418178 0.9218625 0.9757058 1.0000000 0.9932346 0.9558303 0.8901126
## [2,] 0.6243512 0.7405486 0.8418178 0.9218625 0.9757058 1.0000000 0.9932346 0.9558303 0.8901126 0.8001676
## [3,] 0.7405486 0.8418178 0.9218625 0.9757058 1.0000000 0.9932346 0.9558303 0.8901126 0.8001676 0.6915877
```

``` r
summary(tst[,10])
```

```
##        t0        
##  Min.   :0.0000  
##  1st Qu.:0.2246  
##  Median :0.5275  
##  Mean   :0.5154  
##  3rd Qu.:0.8174  
##  Max.   :0.9985
```

``` r
plot_ts(y=ts[,10]) + theme(text = element_text(size=16))
```

![plot of chunk unnamed-chunk-6](fig/ts_norm_gminmax/unnamed-chunk-6-1.png)

