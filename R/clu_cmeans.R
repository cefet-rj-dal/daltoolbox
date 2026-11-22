#'@title Fuzzy c-means
#'@description Fuzzy c-means clustering.
#'@details Produces soft membership for each cluster. The hard assignment is returned by `cluster()`.
#' Membership degrees are returned in the `membership` attribute. The algorithm alternates
#' between updating cluster centers as membership-weighted means and recomputing memberships
#' from the distances to those centers, until convergence. With `nstart > 1` it runs several
#' times from different random starts and keeps the run with the lowest within-cluster fuzzy
#' error, reducing sensitivity to the random start.
#'
#' The base `clusterer()` uses `wcss` as a generic default, but
#' `cluster_cmeans()` specializes that choice because fuzzy clustering carries
#' both a hard partition view and a soft membership structure.
#'
#' Default evaluation in `cluster_cmeans()` is:
#'
#' - main metric: `metric_silhouette()`
#' - internal evaluation: `silhouette`, `withinerror`
#' - external evaluation: `entropy`, `purity`, `adjusted_rand_index`
#'
#' The generic `wcss` fallback is not used as the main metric because fuzzy
#' c-means is better summarized by a partition quality index plus a criterion
#' that respects the model's own within-cluster objective.
#'@param centers number of clusters
#'@param m fuzziness parameter (m > 1)
#'@param iter maximum number of iterations
#'@param dist distance method: "euclidean" or "manhattan"
#'@param tol convergence tolerance on the membership change
#'@param nstart number of restarts; the lowest-error run is kept (default 1)
#'@param seed optional seed for reproducibility
#'@return returns a fuzzy clustering object.
#'@references
#' Bezdek, J. C. (1981). Pattern Recognition with Fuzzy Objective Function Algorithms.
#' Dunn, J. C. (1973). A Fuzzy Relative of the ISODATA Process and Its Use in Detecting Compact Well-Separated Clusters.
#' Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.
#'@examples
#'data(iris)
#'model <- cluster_cmeans(centers = 3, m = 2)
#'model <- fit(model, iris[,1:4])
#'clu <- cluster(model, iris[,1:4])
#'table(clu)
#'@export
cluster_cmeans <- function(centers = 2, m = 2, iter = 100, dist = "euclidean", tol = 1e-6, nstart = 1, seed = NULL) {
  obj <- clusterer()
  utils <- obj$clu_utils
  obj$centers <- centers
  obj$m <- m
  obj$iter <- iter
  obj$dist <- dist
  obj$tol <- tol
  obj$nstart <- nstart
  obj$seed <- seed
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

# Distance from each row of x to each center (n x centers); euclidean or manhattan.
cmeans_distance <- function(x, centers, dist) {
  n <- nrow(x)
  c <- nrow(centers)
  d <- matrix(0, nrow = n, ncol = c)
  for (j in seq_len(c)) {
    diff <- sweep(x, 2, centers[j, ], "-")
    d[, j] <- if (dist == "manhattan") rowSums(abs(diff)) else sqrt(rowSums(diff^2))
  }
  d
}

# One fuzzy c-means run from a random membership init.
cmeans_run <- function(x, centers, m, iter, dist, tol) {
  n <- nrow(x)
  u <- matrix(stats::runif(n * centers), nrow = n, ncol = centers)
  u <- u / rowSums(u)
  power <- 2 / (m - 1)
  d <- NULL
  ctr <- NULL
  for (it in seq_len(iter)) {
    um <- u^m
    ctr <- (t(um) %*% x) / colSums(um)
    d <- cmeans_distance(x, ctr, dist)
    d[d < 1e-10] <- 1e-10
    inv <- 1 / (d^power)
    u_new <- inv / rowSums(inv)
    if (sqrt(sum((u_new - u)^2)) < tol) {
      u <- u_new
      break
    }
    u <- u_new
  }
  list(centers = ctr, membership = u, error = sum((u^m) * (d^2)))
}

#'@importFrom stats runif
#'@exportS3Method fit cluster_cmeans
fit.cluster_cmeans <- function(obj, data, ...) {
  prepared <- clusterer_prepare_fit(obj, data)
  obj <- prepared$obj
  data <- prepared$data
  x <- as.matrix(data)

  tol <- if (is.null(obj$tol)) 1e-6 else obj$tol
  nstart <- if (is.null(obj$nstart)) 1 else max(1, obj$nstart)
  if (!is.null(obj$seed)) set.seed(obj$seed)

  best <- NULL
  for (s in seq_len(nstart)) {
    run <- cmeans_run(x, obj$centers, obj$m, obj$iter, obj$dist, tol)
    if (is.null(best) || run$error < best$error) {
      best <- run
    }
  }
  cluster <- max.col(best$membership, ties.method = "first")
  obj$model <- list(
    centers = best$centers,
    membership = best$membership,
    cluster = cluster,
    withinerror = best$error
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
