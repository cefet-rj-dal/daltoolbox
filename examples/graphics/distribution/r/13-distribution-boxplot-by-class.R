source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)
library(ggplot2)
library(RColorBrewer)

colors <- brewer.pal(3, "Set1")

grf <- plot_boxplot_class(
  datasets::iris[, c("Sepal.Width", "Species")],
  class_label = "Species",
  colors = colors
)
plot(grf)
