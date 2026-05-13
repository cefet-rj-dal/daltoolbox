#'@title Gaussian mixture model clustering (GMM)
#'@description Model-based clustering using `mclust::Mclust`.
#'@details Fits a Gaussian mixture model and returns the MAP classification.
#' The fitted model is stored in `obj$model`. Requires the `mclust` package.
#'@param G number of mixture components (clusters). If NULL, `Mclust` chooses.
#'@param modelNames optional character vector of model names passed to `Mclust`.
#'@return returns a GMM clustering object.
#'@references
#' Fraley, C., & Raftery, A. E. (2002). Model-based clustering. *JASA*.
#'@examples
#'if (requireNamespace("mclust", quietly = TRUE)) {
#'  data(iris)
#'  model <- cluster_gmm(G = 3)
#'  model <- fit(model, iris[,1:4])
#'  clu <- cluster(model, iris[,1:4])
#'  table(clu)
#'}
#'@export
cluster_gmm <- function(G = NULL, modelNames = NULL) {
  obj <- clusterer()
  utils <- obj$clu_utils
  obj$G <- G
  obj$modelNames <- modelNames
  obj$model <- NULL
  obj$metric <- utils$metric_loglik
  obj$metric_name <- "loglik"
  obj$selector <- utils$selector_best
  obj$selector_name <- "best"
  obj$eval_internal <- list(utils$metric_loglik)
  obj$eval_external <- list(utils$metric_entropy, utils$metric_purity, utils$metric_adjusted_rand_index)
  class(obj) <- append("cluster_gmm", class(obj))
  return(obj)
}

#'@importFrom mclust Mclust
#'@exportS3Method fit cluster_gmm
fit.cluster_gmm <- function(obj, data, ...) {
  if (!requireNamespace("mclust", quietly = TRUE)) {
    stop("Package 'mclust' is required for cluster_gmm.", call. = FALSE)
  }
  prepared <- clusterer_prepare_fit(obj, data)
  obj <- prepared$obj
  data <- prepared$data
  mclust_ns <- asNamespace("mclust")
  obj$model <- eval(
    bquote(Mclust(.(data), G = .(obj$G), modelNames = .(obj$modelNames))),
    envir = mclust_ns
  )
  return(obj)
}

#'@exportS3Method cluster cluster_gmm
cluster.cluster_gmm <- function(obj, data, ...) {
  obj <- clusterer_require_fitted(obj)
  if (!identical(adjust_data.frame(data), obj$train_data)) {
    stop("cluster_gmm does not support clustering new data after fit().", call. = FALSE)
  }
  cluster <- obj$model$classification
  if (!is.null(obj$model$loglik)) {
    cluster <- clusterer_attach_metric(cluster, obj$model$loglik, obj$metric_name)
  }
  return(cluster)
}
