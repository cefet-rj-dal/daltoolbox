source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
x <- iris[, 1:4]
ref <- iris$Species
head(x)

base_model <- cluster_kmeans(k = 2)
base_model$metric <- base_model$clu_utils$metric_silhouette
base_model$selector <- base_model$clu_utils$selector_best
base_model$eval_internal <- list(base_model$clu_utils$metric_silhouette)
base_model$eval_external <- list(base_model$clu_utils$metric_entropy)

model <- clu_tune(base_model, ranges = list(k = 2:10))

set_example_seed()
model <- fit(model, x)
model$k

clu <- cluster(model, x)
table(clu)

eval <- evaluate(model, clu, ref)
eval
