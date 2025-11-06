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

data <- iris |> group_by(Species) |> summarize(Sepal.Length=mean(Sepal.Length), Sepal.Width=mean(Sepal.Width))
head(data)

# Grouped bars

# Organizes data into groups per category, showing two or more bars per group, colored by measure.

# More info: https://en.wikipedia.org/wiki/Bar_chart#Grouped_or_stacked

grf <- plot_groupedbar(data, colors=colors[1:2]) + font
plot(grf)
