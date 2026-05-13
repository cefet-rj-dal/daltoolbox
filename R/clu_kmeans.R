#'@title k-means
#'@description k-means clustering using `stats::kmeans`.
#'@details Partitions data into k clusters minimizing within‑cluster sum of squares. The intrinsic
#' quality metric returned is the total within‑cluster SSE (lower is better).
#'
#' The base `clusterer()` uses `wcss` as a generic default, but
#' `cluster_kmeans()` specializes that choice because k-means is usually
#' interpreted through compactness and separation between centroid-based groups.
#'
#' Default evaluation in `cluster_kmeans()` is:
#'
#' - main metric: `metric_silhouette()`
#' - internal evaluation: `silhouette`, `davies_bouldin`, `calinski_harabasz`
#' - external evaluation: `entropy`, `purity`, `adjusted_rand_index`
#'
#' The general `wcss` fallback is not kept as the main metric here because it
#' decreases mechanically as `k` grows and is therefore weaker as a standalone
#' quality criterion for comparing partitions.
#'@param k the number of clusters to form.
#'@return returns a k-means object.
#'@references
#' MacQueen, J. (1967). Some Methods for classification and Analysis of Multivariate Observations.
#' Lloyd, S. (1982). Least squares quantization in PCM.
#' Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.
#' Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure.
#' Calinski, T., & Harabasz, J. (1974). A dendrite method for cluster analysis.
#'@examples
#'# setup clustering
#'model <- cluster_kmeans(k=3)
#'
#'#load dataset
#'data(iris)
#'
#'# build model
#'model <- fit(model, iris[,1:4])
#'clu <- cluster(model, iris[,1:4])
#'table(clu)
#'
#'# evaluate model using external metric
#'eval <- evaluate(model, clu, iris$Species)
#'eval

#'@export
cluster_kmeans <- function(k = 1) {
  obj <- clusterer()
  utils <- obj$clu_utils
  obj$k <- k
  obj$metric <- utils$metric_silhouette
  obj$metric_name <- "silhouette"
  obj$selector <- utils$selector_best
  obj$selector_name <- "best"
  obj$eval_internal <- list(utils$metric_silhouette, utils$metric_davies_bouldin, utils$metric_calinski_harabasz)
  obj$eval_external <- list(utils$metric_entropy, utils$metric_purity, utils$metric_adjusted_rand_index)
  class(obj) <- append("cluster_kmeans", class(obj))
  return(obj)
}

#'@importFrom stats kmeans
#'@exportS3Method fit cluster_kmeans
fit.cluster_kmeans <- function(obj, data, ...) {
  prepared <- clusterer_prepare_fit(obj, data)
  obj <- prepared$obj
  data <- prepared$data

  x <- as.matrix(data)
  storage.mode(x) <- "double"
  obj$model <- stats::kmeans(x = x, centers = obj$k)
  return(obj)
}

#'@exportS3Method cluster cluster_kmeans
cluster.cluster_kmeans <- function(obj, data, ...) {
  obj <- clusterer_require_fitted(obj)
  x <- clusterer_prepare_cluster_data(obj, data)
  x <- as.matrix(x)
  storage.mode(x) <- "double"

  if (identical(x, as.matrix(obj$train_data))) {
    cluster <- obj$model$cluster
    metric <- obj$metric(data = obj$train_data, cluster = cluster, obj = obj)$value
    return(clusterer_attach_metric(cluster, metric, obj$metric_name))
  }

  centers <- obj$model$centers
  dmat <- sapply(seq_len(nrow(centers)), function(i) rowSums(sweep(x, 2, centers[i, ], "-")^2))
  cluster <- max.col(-dmat, ties.method = "first")
  cluster <- clusterer_attach_metric(cluster, NA_real_, obj$metric_name)
  return(cluster)
}

