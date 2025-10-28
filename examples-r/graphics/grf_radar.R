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

# iris dataset for the example
head(iris)

library(dplyr)

data <- iris |> dplyr::select(-Species) 
data <- sapply(data, mean)
data <- data.frame(name = names(data), value = data) |> dplyr::arrange(name)

head(data)


# Radar chart

# Graphical method to display multivariate data with 3+ quantitative variables on axes starting from the same point.

# More info: https://en.wikipedia.org/wiki/Radar_chart

grf <- plot_radar(data, colors=colors[1]) + font
grf <- grf + ylim(0, NA)
plot(grf)
