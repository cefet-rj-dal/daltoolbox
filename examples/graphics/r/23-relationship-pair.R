source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "GGally"))

library(daltoolbox)

if (requireNamespace("GGally", quietly = TRUE)) {
  grf <- plot_pair(
    datasets::iris,
    cnames = colnames(datasets::iris)[1:4],
    title = "Iris scatter matrix",
    clabel = "Species"
  )
  print(grf)
}
