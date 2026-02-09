#'@title Louvain community detection
#'@description Graph community detection using `igraph::cluster_louvain`.
#'@details Accepts an `igraph` object and returns community memberships.
#' Requires the `igraph` package.
#'@param weights optional edge weights to pass to `cluster_louvain`
#'@return returns a Louvain clustering object.
#'@references
#' Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008).
#' Fast unfolding of communities in large networks. *J. Statistical Mechanics*.
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
  obj$weights <- weights
  obj$model <- NULL
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
  obj$model <- igraph::cluster_louvain(data, weights = obj$weights)
  return(obj)
}

#'@exportS3Method cluster cluster_louvain_graph
cluster.cluster_louvain_graph <- function(obj, data, ...) {
  if (is.null(obj$model)) {
    obj <- fit(obj, data)
  }
  cluster <- igraph::membership(obj$model)
  if (!is.null(obj$model$modularity)) {
    attr(cluster, "metric") <- obj$model$modularity
  }
  return(cluster)
}
