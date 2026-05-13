source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation
# install.packages("daltoolbox")

library(daltoolbox)

cluster_agnes_custom <- function(k = 3, method = "ward", dist = "euclidean", scale = TRUE) {
  obj <- daltoolbox::clusterer()
  obj$eval_internal <- list(
    obj$clu_utils$metric_silhouette,
    obj$clu_utils$metric_davies_bouldin
  )
  obj$eval_external <- list(
    obj$clu_utils$metric_entropy
  )
  obj$k <- k
  obj$method <- method
  obj$dist <- dist
  obj$scale <- scale
  class(obj) <- append("cluster_agnes_custom", class(obj))
  obj
}

fit.cluster_agnes_custom <- function(obj, data, ...) {
  obj$train_data <- data
  obj$fitted <- TRUE
  x <- as.matrix(data)
  storage.mode(x) <- "double"

  if (isTRUE(obj$scale)) {
    x <- scale(x)
  }

  obj$model <- cluster::agnes(x, diss = FALSE, metric = obj$dist, method = obj$method)
  obj
}

cluster.cluster_agnes_custom <- function(obj, data, ...) {
  if (!isTRUE(obj$fitted) || is.null(obj$model)) {
    stop("cluster_agnes_custom must be fitted before clustering.", call. = FALSE)
  }

  x <- as.matrix(data)
  storage.mode(x) <- "double"

  clu <- stats::cutree(as.hclust(obj$model), k = obj$k)

  dist <- 0
  for (i in unique(clu)) {
    idx <- i == clu
    center <- colMeans(x[idx, , drop = FALSE])
    dist <- dist + sum(rowSums((x[idx, , drop = FALSE] - center)^2))
  }

  attr(clu, "metric") <- dist
  clu
}

iris <- datasets::iris
model <- cluster_agnes_custom(k = 3, method = "ward")
set_example_seed()
model <- fit(model, iris[, 1:4])
clu <- cluster(model, iris[, 1:4])
table(clu)

eval <- evaluate(model, clu, iris$Species)
eval
