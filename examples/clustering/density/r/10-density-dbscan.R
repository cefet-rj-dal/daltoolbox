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
model <- daltoolbox::fit(model, x)
clu <- daltoolbox::cluster(model, x)
table(clu)

eval <- daltoolbox::evaluate(model, clu, ref)
eval
