source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

hc <- hierarchy_cut(
  "Sepal.Length",
  breaks = c(-Inf, 5.5, 6.5, Inf),
  labels = c("short", "medium", "long")
)

iris_h <- transform(hc, datasets::iris)
table(iris_h$Sepal.Length.Level)
head(iris_h[, c("Sepal.Length", "Sepal.Length.Level")])
