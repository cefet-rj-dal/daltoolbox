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

# Boxplot
# Represents distribution by quartiles; “whiskers” indicate variability outside the quartiles (and help identify outliers).

# More info: https://en.wikipedia.org/wiki/Box_plot

grf <- plot_boxplot(iris, colors="white") + font
plot(grf)  

grf <- plot_boxplot(iris, colors=colors[1:4]) + font
plot(grf)  
