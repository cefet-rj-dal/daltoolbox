# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

library(RColorBrewer)
library(ggplot2)

colors <- brewer.pal(4, 'Set1')

# setting the font size for all charts
font <- theme(text = element_text(size=16))

# conjunto de dados iris para o exemplo
head(iris)

library(dplyr)
data <- iris |> group_by(Species) |> summarize(Sepal.Length=mean(Sepal.Length))
head(data)

# Bar chart

# Displays categorical data with bars proportional to the aggregated value (count, mean, etc.).

# More info: https://en.wikipedia.org/wiki/Bar_chart

grf <- plot_bar(data, colors=colors[1]) + font
plot(grf)

# Bars can be flipped (horizontal/vertical) with coord_flip().
grf <- grf + coord_flip()
plot(grf)

# Bar graph with one color for each species
grf <- plot_bar(data, colors=colors[1:3]) + font
plot(grf)
