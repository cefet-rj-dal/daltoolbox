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

# conjunto de dados iris para o exemplo
head(iris)

library(dplyr)

data <- iris |> group_by(Species) |> summarize(Sepal.Length=mean(Sepal.Length))
head(data)

# Lollipop chart

# Same purpose as the bar chart, highlighting the value with a marker and a stem.

grf <- plot_lollipop(data, colors=colors[1], max_value_gap=0.2) + font
plot(grf)

grf <- plot_lollipop(data, colors=colors[1], max_value_gap=0.1) + font + coord_flip() 
plot(grf)
