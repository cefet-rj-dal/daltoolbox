source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

options(repr.plot.width = 9, repr.plot.height = 4.5)
hc <- hclust(dist(scale(datasets::iris[, 1:4])), method = "ward.D2")
grf <- plot_dendrogram(hc, title = "Iris dendrogram")
plot(grf)
