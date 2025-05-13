# DAL ToolBox
# version 1.2.707



#loading DAL
#library(daltoolbox)

data(sin_data)

library(ggplot2)
plot_ts(x=sin_data$x, y=sin_data$y) + theme(text = element_text(size=16))


sw_size <- 10
ts <- ts_data(sin_data$y, sw_size)
ts_head(ts, 3)

summary(ts[,10])

library(ggplot2)
plot_ts(y=ts[,10]) + theme(text = element_text(size=16))

preproc <- ts_norm_an(outliers = outliers_gaussian())
preproc <- fit(preproc, ts)
tst <- transform(preproc, ts)
ts_head(tst, 3)

summary(as.vector(tst[10,]))

plot_ts(y=tst[,10]) + theme(text = element_text(size=16))

plot_ts(y=tst[10,]) + theme(text = element_text(size=16))

