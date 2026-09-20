source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "GGally", "RColorBrewer"))

library(daltoolbox)
library(GGally)
library(RColorBrewer)

colors <- brewer.pal(3, "Set1")
grf <- plot_pair_adv(
  datasets::iris,
  cnames = colnames(datasets::iris)[1:4],
  title = "Iris advanced pair plot",
  clabel = "Species",
  colors = colors
)
suppressMessages(suppressWarnings(print(grf)))
