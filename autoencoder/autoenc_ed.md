## Vanilla autoencoder transformation (encode-decode)

Considering a dataset with $p$ numerical attributes.

The goal of the autoencoder is to reduce the dimension of $p$ to $k$, such that these $k$ attributes are enough to recompose the original $p$ attributes. However from the $k$ dimensionals the data is returned back to $p$ dimensions. The higher the quality of autoencoder the similiar is the output from the input.


``` r
# DAL ToolBox
# version 1.1.737



#loading DAL
library(daltoolbox)
library(ggplot2)
```

### dataset for example


``` r
data(sin_data)

sw_size <- 5
ts <- ts_data(sin_data$y, sw_size)

ts_head(ts)
```

```
##             t4        t3        t2        t1        t0
## [1,] 0.0000000 0.2474040 0.4794255 0.6816388 0.8414710
## [2,] 0.2474040 0.4794255 0.6816388 0.8414710 0.9489846
## [3,] 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950
## [4,] 0.6816388 0.8414710 0.9489846 0.9974950 0.9839859
## [5,] 0.8414710 0.9489846 0.9974950 0.9839859 0.9092974
## [6,] 0.9489846 0.9974950 0.9839859 0.9092974 0.7780732
```

### applying data normalization


``` r
preproc <- ts_norm_gminmax()
preproc <- fit(preproc, ts)
ts <- transform(preproc, ts)

ts_head(ts)
```

```
##             t4        t3        t2        t1        t0
## [1,] 0.5004502 0.6243512 0.7405486 0.8418178 0.9218625
## [2,] 0.6243512 0.7405486 0.8418178 0.9218625 0.9757058
## [3,] 0.7405486 0.8418178 0.9218625 0.9757058 1.0000000
## [4,] 0.8418178 0.9218625 0.9757058 1.0000000 0.9932346
## [5,] 0.9218625 0.9757058 1.0000000 0.9932346 0.9558303
## [6,] 0.9757058 1.0000000 0.9932346 0.9558303 0.8901126
```

### spliting into training and test


``` r
samp <- ts_sample(ts, test_size = 10)
train <- as.data.frame(samp$train)
test <- as.data.frame(samp$test)
```

### creating autoencoder

Reduce from 5 to 3 dimensions


``` r
auto <- autoenc_ed(5, 3)

auto <- fit(auto, train)
```

### learning curves


``` r
fit_loss <- data.frame(x=1:length(auto$train_loss), train_loss=auto$train_loss,val_loss=auto$val_loss)

grf <- plot_series(fit_loss, colors=c('Blue','Orange'))
plot(grf)
```

![plot of chunk unnamed-chunk-6](fig/autoenc_ed/unnamed-chunk-6-1.png)

### testing autoencoder

presenting the original test set and display encoding


``` r
print(head(test))
```

```
##          t4        t3        t2        t1        t0
## 1 0.7258342 0.8294719 0.9126527 0.9702046 0.9985496
## 2 0.8294719 0.9126527 0.9702046 0.9985496 0.9959251
## 3 0.9126527 0.9702046 0.9985496 0.9959251 0.9624944
## 4 0.9702046 0.9985496 0.9959251 0.9624944 0.9003360
## 5 0.9985496 0.9959251 0.9624944 0.9003360 0.8133146
## 6 0.9959251 0.9624944 0.9003360 0.8133146 0.7068409
```

``` r
result <- transform(auto, test)
```

```
## Called from: eval(expr, p)
## debug at /home/gpca/daltoolbox/R/autoenc_ed.R#51: result <- autoenc_encode_decode(obj$model, data, batch_size = obj$batch_size)
```

``` r
print(head(result))
```

```
##           [,1]      [,2]      [,3]      [,4]      [,5]
## [1,] 0.7217127 0.8282319 0.9116995 0.9669886 0.9985917
## [2,] 0.8280803 0.9119485 0.9693578 0.9979107 0.9959416
## [3,] 0.9154853 0.9717947 0.9989132 0.9977472 0.9621152
## [4,] 0.9758620 1.0020174 0.9972875 0.9660388 0.8996256
## [5,] 0.9979519 0.9949440 0.9610474 0.9034204 0.8135284
## [6,] 0.9918338 0.9598563 0.8978407 0.8158258 0.7073913
```


``` r
result <- as.data.frame(result)
names(result) <- names(test)
r2 <- c()
mape <- c()
for (col in names(test)){
r2_col <- cor(test[col], result[col])^2
r2 <- append(r2, r2_col)
mape_col <- mean((abs((result[col] - test[col]))/test[col])[[col]])
mape <- append(mape, mape_col)
print(paste(col, 'R2 test:', r2_col, 'MAPE:', mape_col))
}
```

```
## [1] "t4 R2 test: 0.999143653193266 MAPE: 0.00297926511681434"
## [1] "t3 R2 test: 0.999722694042756 MAPE: 0.00252325096099102"
## [1] "t2 R2 test: 0.999890214478579 MAPE: 0.00217666393785985"
## [1] "t1 R2 test: 0.999885279399409 MAPE: 0.00397349470656526"
## [1] "t0 R2 test: 0.999987732602016 MAPE: 0.00155169352245455"
```

``` r
print(paste('Means R2 test:', mean(r2), 'MAPE:', mean(mape)))
```

```
## [1] "Means R2 test: 0.999725914743205 MAPE: 0.002640873648937"
```
