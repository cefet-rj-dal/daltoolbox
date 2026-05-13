source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
x <- iris[, 1:4]
ref <- iris$Species

metric_names <- function(model, x, clu, ref = NULL) {
  internal <- vapply(model$eval_internal, function(fn) {
    fn(data = x, cluster = clu, obj = model)$metric
  }, character(1))

  external <- if (is.null(ref)) {
    character(0)
  } else {
    vapply(model$eval_external, function(fn) {
      fn(cluster = clu, attribute = ref, obj = model)$metric
    }, character(1))
  }

  list(internal = internal, external = external)
}

model_default <- cluster_kmeans(k = 3)
set_example_seed()
model_default <- fit(model_default, x)
clu_default <- cluster(model_default, x)

metric_names(model_default, x, clu_default, ref)

eval_default <- evaluate(model_default, clu_default, ref)
eval_default$metrics

model_custom <- cluster_kmeans(k = 3)
model_custom$eval_internal <- list(model_custom$clu_utils$metric_silhouette)
model_custom$eval_external <- list(model_custom$clu_utils$metric_entropy)

set_example_seed()
model_custom <- fit(model_custom, x)
clu_custom <- cluster(model_custom, x)

metric_names(model_custom, x, clu_custom, ref)

eval_custom <- evaluate(model_custom, clu_custom, ref)
eval_custom$metrics

grep("^metric_", names(model_custom$clu_utils), value = TRUE)
