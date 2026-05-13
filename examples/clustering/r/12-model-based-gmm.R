source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "mclust"))

library(daltoolbox)

iris <- datasets::iris
x <- iris[, 1:4]
ref <- iris$Species
head(x)

if (requireNamespace("mclust", quietly = TRUE)) {
  model <- cluster_gmm(G = 3)
}

if (requireNamespace("mclust", quietly = TRUE)) {
  model <- fit(model, x)
  clu <- cluster(model, x)
  table(clu)
}

if (requireNamespace("mclust", quietly = TRUE)) {
  eval <- evaluate(model, clu, ref)
  eval
}
