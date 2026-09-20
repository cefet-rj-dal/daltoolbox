source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

grf <- plot_pixel(as.matrix(datasets::iris[, 1:4]), title = "Iris pixel view")
plot(grf)
