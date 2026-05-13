#'@title Hierarchical clustering
#'@description Agglomerative hierarchical clustering using `stats::hclust`.
#'@details Computes a distance matrix (optionally after scaling) and builds a dendrogram. Clusters are
#' obtained by cutting the tree with `k` (number of clusters) or `h` (height).
#'@param k number of clusters to cut the tree (default 2)
#'@param h height to cut the tree (optional; if provided, overrides `k`)
#'@param method linkage method passed to `stats::hclust` (default "ward.D2")
#'@param dist distance method passed to `stats::dist` (default "euclidean")
#'@param scale logical; whether to scale data before distance (default TRUE)
#'@return returns a hierarchical clustering object.
#'@references
#' Johnson, S. C. (1967). Hierarchical clustering schemes. Psychometrika.
#'@examples
#'data(iris)
#'model <- cluster_hclust(k = 3)
#'model <- fit(model, iris[,1:4])
#'clu <- cluster(model, iris[,1:4])
#'table(clu)
#'@export
cluster_hclust <- function(k = 2, h = NULL,
                           method = c("ward.D2", "ward.D", "single", "complete", "average", "mcquitty", "median", "centroid"),
                           dist = c("euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski"),
                           scale = TRUE) {
  method <- match.arg(method)
  dist <- match.arg(dist)
  obj <- clusterer()
  utils <- obj$clu_utils
  obj$k <- k
  obj$h <- h
  obj$method <- method
  obj$dist <- dist
  obj$scale <- scale
  obj$hc <- NULL
  obj$metric <- utils$metric_silhouette
  obj$metric_name <- "silhouette"
  obj$selector <- utils$selector_best
  obj$selector_name <- "best"
  obj$eval_internal <- list(utils$metric_silhouette, utils$metric_davies_bouldin, utils$metric_calinski_harabasz)
  obj$eval_external <- list(utils$metric_entropy, utils$metric_purity, utils$metric_adjusted_rand_index)
  class(obj) <- append("cluster_hclust", class(obj))
  return(obj)
}

#'@importFrom stats hclust dist
#'@exportS3Method fit cluster_hclust
fit.cluster_hclust <- function(obj, data, ...) {
  prepared <- clusterer_prepare_fit(obj, data)
  obj <- prepared$obj
  data <- prepared$data
  x <- as.matrix(data)
  storage.mode(x) <- "double"
  if (isTRUE(obj$scale)) {
    x <- scale(x)
  }
  d <- stats::dist(x, method = obj$dist)
  obj$hc <- stats::hclust(d, method = obj$method)
  obj$model <- obj$hc
  return(obj)
}

#'@exportS3Method cluster cluster_hclust
cluster.cluster_hclust <- function(obj, data, ...) {
  obj <- clusterer_require_fitted(obj)
  if (!identical(adjust_data.frame(data), obj$train_data)) {
    stop("cluster_hclust does not support clustering new data after fit().", call. = FALSE)
  }
  data_mat <- as.matrix(obj$train_data)
  storage.mode(data_mat) <- "double"
  h_val <- obj$h
  if (is.list(h_val) && length(h_val) == 1) {
    h_val <- h_val[[1]]
  }
  if (!is.null(h_val) && (!is.numeric(h_val) || length(h_val) != 1 || is.na(h_val))) {
    h_val <- NULL
  }

  if (!is.null(h_val)) {
    cluster <- stats::cutree(obj$hc, h = h_val)
  } else {
    cluster <- stats::cutree(obj$hc, k = obj$k)
  }

  metric <- obj$metric(data = obj$train_data, cluster = cluster, obj = obj)$value
  return(clusterer_attach_metric(cluster, metric, obj$metric_name))
}
