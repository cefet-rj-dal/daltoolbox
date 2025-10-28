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
serie <- data.frame(x, sin=sin(x), cosine=cos(x)+5)
head(serie)

# Series chart

# Shows points connected by lines, with x-axis ordered by time/index.

# More info: https://en.wikipedia.org/wiki/Line_chart

grf <- plot_series(serie, colors=colors[1:2]) + font
plot(grf)

  jpeg("series.jpg", width = 640, height = 480)
  plot(grf)
  dev.off()

ggsave("series.png", width = 5, height = 4, units = "cm")
