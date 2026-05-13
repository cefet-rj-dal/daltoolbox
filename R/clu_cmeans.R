#'@title Fuzzy c-means
#'@description Fuzzy c-means clustering using `e1071::cmeans`.
#'@details Produces soft membership for each cluster. The hard assignment is returned by `cluster()`.
#' Membership degrees are returned in the `membership` attribute.
#'@param centers number of clusters
#'@param m fuzziness parameter (m > 1)
#'@param iter maximum number of iterations
#'@param dist distance method passed to `e1071::cmeans`
#'@return returns a fuzzy clustering object.
#'@references
#' Bezdek, J. C. (1981). Pattern Recognition with Fuzzy Objective Function Algorithms.
#'@examples
#'data(iris)
#'model <- cluster_cmeans(centers = 3, m = 2)
#'model <- fit(model, iris[,1:4])
#'clu <- cluster(model, iris[,1:4])
#'table(clu)
#'@export
cluster_cmeans <- function(centers = 2, m = 2, iter = 100, dist = "euclidean") {
  obj <- clusterer()
  utils <- obj$clu_utils
  obj$centers <- centers
  obj$m <- m
  obj$iter <- iter
  obj$dist <- dist
  obj$model <- NULL
  obj$metric <- utils$metric_silhouette
  obj$metric_name <- "silhouette"
  obj$selector <- utils$selector_best
  obj$selector_name <- "best"
  obj$eval_internal <- list(utils$metric_silhouette, utils$metric_withinerror)
  obj$eval_external <- list(utils$metric_entropy, utils$metric_purity, utils$metric_adjusted_rand_index)
  class(obj) <- append("cluster_cmeans", class(obj))
  return(obj)
}

#'@importFrom e1071 cmeans
#'@exportS3Method fit cluster_cmeans
fit.cluster_cmeans <- function(obj, data, ...) {
  prepared <- clusterer_prepare_fit(obj, data)
  obj <- prepared$obj
  data <- prepared$data
  obj$model <- e1071::cmeans(
    data,
    centers = obj$centers,
    m = obj$m,
    iter.max = obj$iter,
    dist = obj$dist
  )
  return(obj)
}

#'@exportS3Method cluster cluster_cmeans
cluster.cluster_cmeans <- function(obj, data, ...) {
  obj <- clusterer_require_fitted(obj)
  if (!identical(adjust_data.frame(data), obj$train_data)) {
    stop("cluster_cmeans does not support clustering new data after fit().", call. = FALSE)
  }
  model <- obj$model
  cluster <- model$cluster
  attr(cluster, "membership") <- model$membership
  metric <- obj$metric(data = obj$train_data, cluster = cluster, obj = obj)$value
  cluster <- clusterer_attach_metric(cluster, metric, obj$metric_name)
  return(cluster)
}
