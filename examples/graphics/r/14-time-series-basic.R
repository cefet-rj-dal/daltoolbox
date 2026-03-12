# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

library(ggplot2)
library(RColorBrewer)

# color palette
colors <- brewer.pal(4, 'Set1')

# setting the font size for all charts
font <- theme(text = element_text(size=16))

# Synthetic time series

x <- seq(0, 10, 0.25)
y <- sin(x)

# Time series chart

# Basic exploratory visualization of a time series

grf <- plot_ts(x = x, y = y, color=c("red"))
plot(grf)
