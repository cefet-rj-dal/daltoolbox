source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
x <- iris[, 1:4]
ref <- iris$Species
head(x)

model <- cluster_dbscan(minPts = 3)
model$eval_external <- list(
  model$clu_utils$metric_entropy,
  model$clu_utils$metric_purity
)

set_example_seed()
model <- fit(model, x)
clu <- cluster(model, x)
table(clu)

eval <- evaluate(model, clu, ref)
eval
