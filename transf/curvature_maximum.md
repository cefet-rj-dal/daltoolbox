
``` r
# installation 
install.packages"daltoolbox")

# loading DAL
library(daltoolbox) 

# for ploting
library(ggplot2)
library(dplyr)
```


``` r
# Maximum curvature
# If the curve is decreasing, use maximum curvature analysis. 
# It brings a trade-off between having lower x values (with not so low y values) and having higher x values (not having to much decrease in y values). 

x <- seq(from=1,to=10,by=0.5)
dat <- data.frame(x = x, value = -log(x), variable = as.factor("log"))
myfit <- fit_curvature_max()
res <- transform(myfit, dat$value)
head(res)
```

```
##   x         y         yfit
## 1 9 -1.609438 9.224359e-08
```


``` r
grf <- plot_scatter(dat, label_x = "dimensions", label_y = "cumulative variance", colors="black") + 
    theme(text = element_text(size=16))
plot(grf + geom_vline(xintercept = dat$x[res$x], linetype="dashed", color = "red", size=0.5))
```

![plot of chunk unnamed-chunk-3](fig/curvature_maximum/unnamed-chunk-3-1.png)

