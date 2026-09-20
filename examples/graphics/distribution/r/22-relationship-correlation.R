source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

grf <- plot_correlation(datasets::iris[, 1:4], reorder = "hclust")
plot(grf)
