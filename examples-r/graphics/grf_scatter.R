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

# Scatter plot

# Used to visualize the relationship between two numeric variables.
# The first selected column is treated as X (independent) and the second as Y (dependent);
# a third categorical variable can color the points.

# The color vector must match the number of levels/groups.

# More info: https://en.wikipedia.org/wiki/Scatter_plot

library(dplyr)

grf <- plot_scatter(
  iris |> dplyr::select(x = Sepal.Length, value = Sepal.Width, variable = Species),
  label_x = "Sepal.Length",  # X-axis label
  label_y = "Sepal.Width",   # Y-axis label
  colors=colors[1:3]          # one color per Species level
) + font
plot(grf)
