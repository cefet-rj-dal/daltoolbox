# installation
# install.packages("daltoolbox")

library(daltoolbox)

cluster_agnes_custom <- function(k = 3, method = "ward", metric = "euclidean", scale = TRUE) {
  obj <- daltoolbox::clusterer()
  obj$k <- k
  obj$method <- method
  obj$metric <- metric
  obj$scale <- scale
  class(obj) <- append("cluster_agnes_custom", class(obj))
  obj
}

fit.cluster_agnes_custom <- function(obj, data, ...) {
  x <- as.matrix(data)
  storage.mode(x) <- "double"

  if (isTRUE(obj$scale)) {
    x <- scale(x)
  }

  obj$model <- cluster::agnes(x, diss = FALSE, metric = obj$metric, method = obj$method)
  obj
}

cluster.cluster_agnes_custom <- function(obj, data, ...) {
  if (is.null(obj$model)) {
    obj <- fit(obj, data)
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
model <- fit(model, iris[, 1:4])
clu <- cluster(model, iris[, 1:4])
table(clu)

eval <- evaluate(model, clu, iris$Species)
eval
