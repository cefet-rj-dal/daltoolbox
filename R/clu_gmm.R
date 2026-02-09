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
#'data(iris)
#'model <- cluster_gmm(G = 3)
#'model <- fit(model, iris[,1:4])
#'clu <- cluster(model, iris[,1:4])
#'table(clu)
#'@export
cluster_gmm <- function(G = NULL, modelNames = NULL) {
  obj <- clusterer()
  obj$G <- G
  obj$modelNames <- modelNames
  obj$model <- NULL
  class(obj) <- append("cluster_gmm", class(obj))
  return(obj)
}

#'@importFrom mclust Mclust
#'@importFrom utils getFromNamespace
#'@exportS3Method fit cluster_gmm
fit.cluster_gmm <- function(obj, data, ...) {
  f <- getFromNamespace("Mclust", "mclust")
  obj$model <- f(data, G = obj$G, modelNames = obj$modelNames)
  return(obj)
}

#'@exportS3Method cluster cluster_gmm
cluster.cluster_gmm <- function(obj, data, ...) {
  if (is.null(obj$model)) {
    obj <- fit(obj, data)
  }
  cluster <- obj$model$classification
  if (!is.null(obj$model$loglik)) {
    attr(cluster, "metric") <- obj$model$loglik
  }
  return(cluster)
}
