#'@title Louvain community detection
#'@description Graph community detection using `igraph::cluster_louvain`.
#'@details Accepts an `igraph` object and returns community memberships.
#' Requires the `igraph` package.
#'
#' The base `clusterer()` uses `wcss` and `entropy` as generic defaults, but
#' `cluster_louvain_graph()` specializes both because graph community detection
#' is not a point-to-centroid clustering problem.
#'
#' Default evaluation in `cluster_louvain_graph()` is:
#'
#' - main metric: `metric_modularity()`
#' - internal evaluation: `modularity`
#' - external evaluation: none
#'
#' The general `wcss` fallback is not used because graph communities are judged
#' by network connectivity structure, not by geometric dispersion. The generic
#' external `entropy` default is also not kept because this class operates on an
#' `igraph` object and has no natural reference labels unless the user supplies
#' an external comparison separately.
#'@param weights optional edge weights to pass to `cluster_louvain`
#'@return returns a Louvain clustering object.
#'@references
#' Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008).
#' Fast unfolding of communities in large networks. *J. Statistical Mechanics*.
#' Newman, M. E. J. (2006). Modularity and community structure in networks.
#'@examples
#'if (requireNamespace("igraph", quietly = TRUE)) {
#'  g <- igraph::sample_gnp(n = 20, p = 0.15)
#'  model <- cluster_louvain_graph()
#'  model <- fit(model, g)
#'  clu <- cluster(model, g)
#'  table(clu)
#'}
#'@export
cluster_louvain_graph <- function(weights = NULL) {
  obj <- clusterer()
  utils <- obj$clu_utils
  obj$weights <- weights
  obj$model <- NULL
  obj$metric <- utils$metric_modularity
  obj$metric_name <- "modularity"
  obj$selector <- utils$selector_best
  obj$selector_name <- "best"
  obj$eval_internal <- list(utils$metric_modularity)
  obj$eval_external <- list()
  class(obj) <- append("cluster_louvain_graph", class(obj))
  return(obj)
}

#'@exportS3Method fit cluster_louvain_graph
fit.cluster_louvain_graph <- function(obj, data, ...) {
  if (!requireNamespace("igraph", quietly = TRUE)) {
    stop("cluster_louvain_graph requer o pacote 'igraph'. Instale com install.packages('igraph').")
  }
  if (!igraph::is_igraph(data)) {
    stop("cluster_louvain_graph: 'data' deve ser um objeto igraph.")
  }
  prepared <- clusterer_prepare_fit(obj, data)
  obj <- prepared$obj
  data <- prepared$data
  obj$model <- igraph::cluster_louvain(data, weights = obj$weights)
  return(obj)
}

#'@exportS3Method cluster cluster_louvain_graph
cluster.cluster_louvain_graph <- function(obj, data, ...) {
  obj <- clusterer_require_fitted(obj)
  if (!identical(data, obj$train_data)) {
    stop("cluster_louvain_graph does not support clustering a new graph after fit().", call. = FALSE)
  }
  cluster <- igraph::membership(obj$model)
  if (!is.null(obj$model$modularity)) {
    cluster <- clusterer_attach_metric(cluster, obj$model$modularity, obj$metric_name)
  }
  return(cluster)
}
