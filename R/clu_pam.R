#'@title PAM (Partitioning Around Medoids)
#'@description Clustering around representative data points (medoids), using a native
#' implementation of PAM (no external clustering dependency).
#'@details More robust to outliers than k‑means. The intrinsic metric reported is the within‑cluster SSE to medoids.
#'
#' PAM proceeds in two phases: BUILD, a greedy selection of the initial medoids, and
#' SWAP, which exchanges a medoid with a non-medoid whenever that lowers the total
#' distance to the nearest medoid. The SWAP phase uses the FastPAM1 scheme
#' (Schubert & Rousseeuw, 2019), which evaluates the cost change of every candidate swap
#' in a single pass and therefore reaches the same local optimum as classic PAM (identical
#' medoids) while being markedly faster as the number of clusters grows (e.g. about 20x at
#' k = 20 in the reference port). This is an independent reimplementation (inspired by
#' `cluster::pam`, but not a wrapper of it), so the package carries no dependency on the
#' `cluster` package.
#'
#' The base `clusterer()` uses `wcss` as a generic default, but
#' `cluster_pam()` specializes that choice because PAM is commonly compared
#' through partition quality indices that balance cohesion and separation.
#'
#' Default evaluation in `cluster_pam()` is:
#'
#' - main metric: `metric_silhouette()`
#' - internal evaluation: `silhouette`, `davies_bouldin`, `calinski_harabasz`
#' - external evaluation: `entropy`, `purity`, `adjusted_rand_index`
#'
#' The generic `wcss` fallback is not used as the main metric because PAM is not
#' optimized around centroids and is usually interpreted through medoid-based
#' partition quality rather than only through within-cluster dispersion.
#'@param k the number of clusters to generate.
#'@return returns PAM object.
#'@references
#' Kaufman, L. and Rousseeuw, P. J. (1990). Finding Groups in Data: An Introduction to Cluster Analysis.
#' Schubert, E. and Rousseeuw, P. J. (2019). Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms. In Similarity Search and Applications (SISAP 2019), 171-187.
#' Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.
#' Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure.
#' Calinski, T., & Harabasz, J. (1974). A dendrite method for cluster analysis.
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

# PAM BUILD: greedy selection of the initial k medoids from a distance matrix.
pam_build <- function(d, k) {
  n <- nrow(d)
  medoids <- which.min(rowSums(d))
  while (length(medoids) < k) {
    nearest <- apply(d[, medoids, drop = FALSE], 1, min)
    # candidate h in rows, point o in columns: gain of adding h = sum_o max(nearest_o - d[h,o], 0)
    nmat <- matrix(nearest, nrow = n, ncol = n, byrow = TRUE)
    gains <- rowSums(pmax(nmat - d, 0))
    gains[medoids] <- -Inf
    medoids <- c(medoids, which.max(gains))
  }
  sort(medoids)
}

# PAM SWAP (FastPAM1): one vectorized pass evaluates every medoid/non-medoid swap.
pam_swap <- function(d, medoids, max_iter = 100) {
  n <- nrow(d)
  k <- length(medoids)
  if (k < 2) return(medoids)
  for (it in seq_len(max_iter)) {
    dm <- d[, medoids, drop = FALSE]                       # n x k
    ord <- t(apply(dm, 1, order))
    nearest_slot <- ord[, 1]
    d1 <- dm[cbind(seq_len(n), ord[, 1])]
    d2 <- dm[cbind(seq_len(n), ord[, 2])]
    cand <- setdiff(seq_len(n), medoids)
    if (length(cand) == 0) break
    dc <- d[, cand, drop = FALSE]                          # n x c
    a <- pmin(d1, dc); dim(a) <- dim(dc); a <- a - d1      # change when removed medoid != nearest
    b <- pmin(d2, dc); dim(b) <- dim(dc); b <- b - d1      # change when removed medoid == nearest
    shared <- colSums(a)
    ba <- b - a
    corr <- matrix(0, nrow = k, ncol = length(cand))
    for (i in seq_len(k)) {
      rows_i <- which(nearest_slot == i)
      if (length(rows_i)) corr[i, ] <- colSums(ba[rows_i, , drop = FALSE])
    }
    deltas <- sweep(corr, 2, shared, "+")                  # k x c = delta TD(slot i, candidate j)
    idx <- which.min(deltas)
    bi <- ((idx - 1) %% k) + 1
    bc <- ((idx - 1) %/% k) + 1
    if (deltas[bi, bc] >= -1e-9) break
    medoids[bi] <- cand[bc]
    medoids <- sort(medoids)
  }
  medoids
}

# Native PAM: returns a model list compatible with the cluster.cluster_pam method.
pam_native <- function(data, k, max_iter = 100) {
  x <- as.matrix(data)
  d <- as.matrix(stats::dist(x, method = "euclidean"))
  medoids <- pam_build(d, k)
  medoids <- pam_swap(d, medoids, max_iter)
  clustering <- max.col(-d[, medoids, drop = FALSE], ties.method = "first")
  list(
    medoids = x[medoids, , drop = FALSE],
    id.med = medoids,
    clustering = clustering
  )
}

#'@importFrom stats dist
#'@exportS3Method fit cluster_pam
fit.cluster_pam <- function(obj, data, ...) {
  prepared <- clusterer_prepare_fit(obj, data)
  obj <- prepared$obj
  data <- prepared$data

  obj$model <- pam_native(data, obj$k)
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

