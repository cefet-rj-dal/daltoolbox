#'@title PAM (Partitioning Around Medoids)
#'@description Clustering around representative data points (medoids) using `cluster::pam`.
#'@details More robust to outliers than k‑means. The intrinsic metric reported is the within‑cluster SSE to medoids.
#'@param k the number of clusters to generate.
#'@return returns PAM object.
#'@references
#' Kaufman, L. and Rousseeuw, P. J. (1990). Finding Groups in Data: An Introduction to Cluster Analysis.
#'@examples
#'# setup clustering
#'model <- cluster_pam(k = 3)
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
cluster_pam <- function(k = 1) {
  obj <- clusterer()
  utils <- obj$clu_utils
  obj$k <- k
  obj$metric <- utils$metric_silhouette
  obj$metric_name <- "silhouette"
  obj$selector <- utils$selector_best
  obj$selector_name <- "best"
  obj$eval_internal <- list(utils$metric_silhouette, utils$metric_davies_bouldin, utils$metric_calinski_harabasz)
  obj$eval_external <- list(utils$metric_entropy, utils$metric_purity, utils$metric_adjusted_rand_index)

  class(obj) <- append("cluster_pam", class(obj))
  return(obj)
}

#'@importFrom cluster pam
#'@exportS3Method fit cluster_pam
fit.cluster_pam <- function(obj, data, ...) {
  prepared <- clusterer_prepare_fit(obj, data)
  obj <- prepared$obj
  data <- prepared$data

  obj$model <- cluster::pam(data, obj$k)
  return(obj)
}

#'@exportS3Method cluster cluster_pam
cluster.cluster_pam <- function(obj, data, ...) {
  obj <- clusterer_require_fitted(obj)
  x <- clusterer_prepare_cluster_data(obj, data)

  if (identical(x, obj$train_data)) {
    cluster <- obj$model$clustering
    metric <- obj$metric(data = obj$train_data, cluster = cluster, obj = obj)$value
    return(clusterer_attach_metric(cluster, metric, obj$metric_name))
  }

  x <- as.matrix(x)
  medoids <- as.matrix(obj$model$medoids)
  dmat <- sapply(seq_len(nrow(medoids)), function(i) rowSums(sweep(x, 2, medoids[i, ], "-")^2))
  cluster <- max.col(-dmat, ties.method = "first")
  cluster <- clusterer_attach_metric(cluster, NA_real_, obj$metric_name)
  return(cluster)
}

