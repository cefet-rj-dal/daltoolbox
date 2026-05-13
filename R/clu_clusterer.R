#'@title Clusterer
#'@description Base class for clustering algorithms and related evaluation utilities.
#'
#'@details
#' The object stores shared state and defaults used by clustering methods.
#' Current algorithms may still differ in how much they use this state, but the
#' goal is to standardize future implementations around:
#'
#' - `fit()` learning and storing model state
#' - `cluster()` producing labels from a fitted model
#' - configurable internal/external metrics and selection helpers via `cluutils()`
#'
#' The base defaults are intentionally generic:
#'
#' - `metric = metric_wcss()`
#' - `eval_internal = list(metric_wcss)`
#' - `eval_external = list(metric_entropy)`
#'
#' These defaults provide a common contract for simple partition-based clustering,
#' but they are not imposed on all subclasses. Concrete clustering methods can
#' replace them when another metric is more faithful to the algorithm. For
#' example, centroid and medoid methods often prefer separation-based indices
#' such as silhouette, density-based methods may prefer counting noise points,
#' mixture models naturally expose log-likelihood, and graph community detection
#' is better described by modularity.
#'
#' Therefore, users should treat the `clusterer()` defaults as a shared fallback,
#' not as the final evaluation policy for every `cluster_*` implementation.
#'@return returns a `clusterer` object
#'@examples
#'#See ?cluster_kmeans for an example of transformation
#'@export
clusterer <- function() {
  obj <- dal_learner()
  utils <- cluutils()
  class(obj) <- append("clusterer", class(obj))
  obj$model <- NULL
  obj$train_data <- NULL
  obj$xnames <- NULL
  obj$fitted <- FALSE
  obj$clu_utils <- utils
  obj$metric <- utils$metric_wcss
  obj$metric_name <- "wcss"
  obj$selector <- utils$selector_best
  obj$selector_name <- "best"
  obj$eval_internal <- list(utils$metric_wcss)
  obj$eval_external <- list(utils$metric_entropy)
  return(obj)
}

clusterer_prepare_fit <- function(obj, data) {
  if (is.data.frame(data) || is.matrix(data)) {
    data <- adjust_data.frame(data)
  }
  obj$train_data <- data
  obj$xnames <- if (is.data.frame(data) || is.matrix(data)) colnames(data) else NULL
  obj$fitted <- TRUE
  return(list(obj = obj, data = data))
}

clusterer_prepare_cluster_data <- function(obj, data) {
  if (is.data.frame(data) || is.matrix(data)) {
    data <- adjust_data.frame(data)
  }
  if (!is.null(obj$xnames) && (is.data.frame(data) || is.matrix(data))) {
    common <- intersect(obj$xnames, colnames(data))
    if (length(common) == length(obj$xnames)) {
      data <- data[, obj$xnames, drop = FALSE]
    }
  }
  return(data)
}

clusterer_require_fitted <- function(obj) {
  if (!isTRUE(obj$fitted) || is.null(obj$model)) {
    stop(sprintf("%s must be fitted before clustering.", class(obj)[1]), call. = FALSE)
  }
  return(obj)
}

clusterer_metric_value <- function(value) {
  if (is.null(value) || length(value) == 0) {
    return(NA_real_)
  }
  value <- as.numeric(value)
  value <- value[!is.na(value)]
  if (length(value) == 0) {
    return(NA_real_)
  }
  value[length(value)]
}

clusterer_attach_metric <- function(cluster, value, metric) {
  value <- clusterer_metric_value(value)
  if (!is.na(value)) {
    attr(cluster, "metric") <- value
  }
  attr(cluster, "metric_name") <- metric
  return(cluster)
}

clusterer_normalize_metrics <- function(obj, metrics) {
  utils <- if (!is.null(obj$clu_utils)) obj$clu_utils else cluutils()

  if (is.null(metrics)) {
    return(list())
  }

  if (is.function(metrics)) {
    return(list(metrics))
  }

  if (is.character(metrics)) {
    out <- vector("list", length(metrics))
    for (i in seq_along(metrics)) {
      ref <- utils[[metrics[i]]]
      if (!is.function(ref)) {
        stop(sprintf("Unknown clustering metric '%s'.", metrics[i]), call. = FALSE)
      }
      out[[i]] <- ref
    }
    return(out)
  }

  metrics
}

#'@exportS3Method action clusterer
action.clusterer <- function(obj, ...) {
  thiscall <- match.call(expand.dots = TRUE)
  # proxy action() to cluster() for clusterers
  thiscall[[1]] <- as.name("cluster")
  result <- eval.parent(thiscall)
  return(result)
}

#'@title Cluster
#'@description Generic for clustering methods
#'@param obj a `clusterer` object
#'@param ... optional arguments
#'@return clustered data
#'@examples
#'#See ?cluster_kmeans for an example of transformation
#' @export
cluster <- function(obj, ...) {
  UseMethod("cluster")
}

#'@exportS3Method cluster default
cluster.default <- function(obj, ...) {
  return(data.frame())
}

#'@importFrom dplyr filter summarise group_by n
#'@exportS3Method evaluate clusterer
evaluate.clusterer <- function(obj, cluster, attribute, ...) {
  utils <- if (!is.null(obj$clu_utils)) obj$clu_utils else cluutils()
  result <- list()
  metric_rows <- NULL
  data <- obj$train_data
  internal_metrics <- clusterer_normalize_metrics(obj, obj$eval_internal)
  external_metrics <- clusterer_normalize_metrics(obj, obj$eval_external)

  for (metric_fn in internal_metrics) {
    metric_res <- metric_fn(data = data, cluster = cluster, obj = obj)
    metric_rows <- rbind(metric_rows, data.frame(
      metric = metric_res$metric,
      value = metric_res$value,
      goal = metric_res$goal,
      type = metric_res$type,
      check.names = FALSE
    ))
  }

  if (!missing(attribute)) {
    for (metric_fn in external_metrics) {
      metric_res <- metric_fn(cluster = cluster, attribute = attribute, obj = obj)
      metric_rows <- rbind(metric_rows, data.frame(
        metric = metric_res$metric,
        value = metric_res$value,
        goal = metric_res$goal,
        type = metric_res$type,
        check.names = FALSE
      ))

      if (identical(metric_res$metric, "entropy")) {
        result$clusters_entropy <- metric_res$details
        result$clustering_entropy <- metric_res$value
        basic <- utils$metric_entropy(as.factor(rep(1, length(attribute))), attribute)
        result$data_entropy <- basic$value
      }
    }
  }

  result$metrics <- metric_rows

  return(result)
}
