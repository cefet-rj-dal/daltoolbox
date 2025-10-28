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

# Pie chart
# Circular chart divided into slices to illustrate proportions.

# More info: https://en.wikipedia.org/wiki/Pie_chart

grf <- plot_pieplot(data, colors=colors[1:3]) + font
plot(grf)
