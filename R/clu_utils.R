#'@title Clustering utilities
#'@description Utility object that groups clustering metrics and model-selection helpers.
#'
#'@details
#' The object organizes helpers into two semantic groups:
#'
#' \strong{Metrics}
#'
#' - `metric_wcss()` computes the total within-cluster sum of squares.
#' - `metric_silhouette()` computes the mean silhouette score from pairwise distances.
#' - `metric_entropy()` computes external clustering entropy against a reference label.
#' - `metric_purity()` computes cluster purity against a reference label.
#' - `metric_davies_bouldin()` computes the Davies-Bouldin index.
#' - `metric_calinski_harabasz()` computes the Calinski-Harabasz score.
#' - `metric_adjusted_rand_index()` computes the adjusted Rand index.
#' - `metric_noise_points()` summarizes the number of noise points in density-based clustering.
#' - `metric_loglik()` and `metric_modularity()` expose model-specific quality summaries.
#'
#' \strong{Selectors}
#'
#' - `selector_best()` selects the best hyperparameter value by direct optimization.
#' - `selector_elbow()` selects the elbow of a metric curve via maximum curvature.
#'
#' Metric helpers return a standardized list with fields `metric`, `value`, `goal`,
#' and `type`. This keeps the contract uniform even when the metrics themselves differ.
#'
#'@return returns a `cluutils` object exposing metric and selector helpers.
#'@examples
#'utils <- cluutils()
#'
#'data(iris)
#'x <- iris[, 1:4]
#'clu <- stats::kmeans(x, centers = 3)$cluster
#'
#'utils$metric_wcss(x, clu)
#'utils$metric_silhouette(x, clu)
#'utils$metric_entropy(clu, iris$Species)
#'utils$selector_best(c(0.31, 0.42, 0.39), goal = "maximize")
#'@export
cluutils <- function() {
  metric_result <- function(metric, value, goal, type, details = NULL) {
    result <- list(metric = metric, value = as.numeric(value), goal = goal, type = type)
    if (!is.null(details)) {
      result$details <- details
    }
    return(result)
  }

  cluster_centers <- function(data, cluster) {
    x <- as.matrix(data)
    storage.mode(x) <- "double"
    labels <- as.integer(as.factor(cluster))
    centers <- lapply(unique(labels), function(i) {
      idx <- labels == i
      colMeans(x[idx, , drop = FALSE])
    })
    centers <- do.call(rbind, centers)
    rownames(centers) <- unique(labels)
    list(data = x, labels = labels, centers = centers)
  }

  metric_from_attr <- function(cluster, metric, goal, type = "model", default = NA_real_, obj = NULL) {
    value <- NULL
    if (!is.null(obj) && !is.null(obj$model) && !is.null(obj$model[[metric]])) {
      value <- obj$model[[metric]]
    }
    if (is.null(value)) {
      value <- attr(cluster, "metric", exact = TRUE)
    }
    if (!is.null(value) && length(value) > 1) {
      value <- as.numeric(value)
      value <- value[!is.na(value)]
      value <- if (length(value) == 0) default else value[length(value)]
    }
    if (is.null(value) || length(value) == 0) value <- default
    metric_result(metric, value, goal, type)
  }

  metric_wcss <- function(data, cluster, ...) {
    x <- as.matrix(data)
    storage.mode(x) <- "double"
    labels <- as.integer(as.factor(cluster))
    value <- 0
    for (i in unique(labels)) {
      idx <- labels == i
      if (!any(idx)) next
      center <- colMeans(x[idx, , drop = FALSE])
      value <- value + sum(rowSums((x[idx, , drop = FALSE] - center)^2))
    }
    metric_result("wcss", value, "minimize", "internal")
  }

  metric_silhouette <- function(data, cluster, ...) {
    x <- as.matrix(data)
    storage.mode(x) <- "double"
    labels <- as.integer(as.factor(cluster))
    d <- as.matrix(stats::dist(x))
    n <- nrow(d)
    sil <- rep(0, n)

    for (i in seq_len(n)) {
      own <- labels[i]
      own_idx <- which(labels == own)
      own_idx <- own_idx[own_idx != i]

      a <- 0
      if (length(own_idx) > 0) {
        a <- mean(d[i, own_idx])
      }

      other_clusters <- setdiff(unique(labels), own)
      if (length(other_clusters) == 0) {
        sil[i] <- 0
        next
      }

      b <- Inf
      for (j in other_clusters) {
        idx <- which(labels == j)
        if (length(idx) == 0) next
        b <- min(b, mean(d[i, idx]))
      }

      if (!is.finite(b) || (a == 0 && b == 0)) {
        sil[i] <- 0
      } else {
        sil[i] <- (b - a) / max(a, b)
      }
    }

    metric_result("silhouette", mean(sil), "maximize", "internal")
  }

  metric_davies_bouldin <- function(data, cluster, ...) {
    parts <- cluster_centers(data, cluster)
    x <- parts$data
    labels <- parts$labels
    centers <- parts$centers
    k <- nrow(centers)
    if (k <= 1) {
      return(metric_result("davies_bouldin", NA_real_, "minimize", "internal"))
    }

    scatters <- rep(0, k)
    for (i in seq_len(k)) {
      idx <- labels == as.integer(rownames(centers)[i])
      if (!any(idx)) next
      scatters[i] <- mean(sqrt(rowSums(sweep(x[idx, , drop = FALSE], 2, centers[i, ], "-")^2)))
    }

    center_dist <- as.matrix(stats::dist(centers))
    db <- rep(0, k)
    for (i in seq_len(k)) {
      ratios <- rep(-Inf, k)
      for (j in seq_len(k)) {
        if (i == j || center_dist[i, j] == 0) next
        ratios[j] <- (scatters[i] + scatters[j]) / center_dist[i, j]
      }
      db[i] <- max(ratios[is.finite(ratios)])
    }

    metric_result("davies_bouldin", mean(db), "minimize", "internal")
  }

  metric_calinski_harabasz <- function(data, cluster, ...) {
    parts <- cluster_centers(data, cluster)
    x <- parts$data
    labels <- parts$labels
    centers <- parts$centers
    n <- nrow(x)
    k <- nrow(centers)
    if (k <= 1 || k >= n) {
      return(metric_result("calinski_harabasz", NA_real_, "maximize", "internal"))
    }

    global_center <- colMeans(x)
    wcss <- metric_wcss(x, labels)$value
    bcss <- 0
    for (i in seq_len(k)) {
      idx <- labels == as.integer(rownames(centers)[i])
      ni <- sum(idx)
      if (ni == 0) next
      bcss <- bcss + ni * sum((centers[i, ] - global_center)^2)
    }

    value <- (bcss / (k - 1)) / (wcss / (n - k))
    metric_result("calinski_harabasz", value, "maximize", "internal")
  }

  metric_entropy <- function(cluster, attribute, ...) {
    x <- as.factor(cluster)
    y <- as.factor(attribute)

    dataset <- data.frame(x = x, y = y)
    value <- getOption("dplyr.summarise.inform")
    options(dplyr.summarise.inform = FALSE)

    qtd <- t <- e <- ce <- ceg <- NULL
    tbl <- dataset |>
      dplyr::group_by(x, y) |>
      dplyr::summarise(qtd = dplyr::n())
    tbs <- dataset |>
      dplyr::group_by(x) |>
      dplyr::summarise(t = dplyr::n())
    tbl <- base::merge(x = tbl, y = tbs, by.x = "x", by.y = "x")
    tbl$e <- -(tbl$qtd / tbl$t) * log(tbl$qtd / tbl$t, 2)
    tbl <- tbl |>
      dplyr::group_by(x) |>
      dplyr::summarise(ce = sum(e), qtd = sum(qtd))
    tbl$ceg <- tbl$ce * tbl$qtd / length(x)

    options(dplyr.summarise.inform = value)

    metric_result("entropy", sum(tbl$ceg), "minimize", "external", details = tbl)
  }

  metric_purity <- function(cluster, attribute, ...) {
    tbl <- table(as.factor(cluster), as.factor(attribute))
    value <- sum(apply(tbl, 1, max)) / sum(tbl)
    metric_result("purity", value, "maximize", "external", details = tbl)
  }

  metric_rand_index <- function(cluster, attribute, ...) {
    x <- as.integer(as.factor(cluster))
    y <- as.integer(as.factor(attribute))
    n <- length(x)
    if (n <= 1) {
      return(metric_result("rand_index", NA_real_, "maximize", "external"))
    }
    agree <- 0
    total <- 0
    for (i in seq_len(n - 1)) {
      for (j in seq.int(i + 1, n)) {
        agree_x <- x[i] == x[j]
        agree_y <- y[i] == y[j]
        agree <- agree + as.integer(agree_x == agree_y)
        total <- total + 1
      }
    }
    metric_result("rand_index", agree / total, "maximize", "external")
  }

  metric_adjusted_rand_index <- function(cluster, attribute, ...) {
    tbl <- table(as.factor(cluster), as.factor(attribute))
    n <- sum(tbl)
    if (n <= 1) {
      return(metric_result("adjusted_rand_index", NA_real_, "maximize", "external"))
    }

    comb2 <- function(x) ifelse(x < 2, 0, x * (x - 1) / 2)
    sum_ij <- sum(comb2(tbl))
    sum_i <- sum(comb2(rowSums(tbl)))
    sum_j <- sum(comb2(colSums(tbl)))
    total <- comb2(n)
    expected <- (sum_i * sum_j) / total
    max_index <- (sum_i + sum_j) / 2
    value <- (sum_ij - expected) / (max_index - expected)
    metric_result("adjusted_rand_index", value, "maximize", "external")
  }

  metric_noise_points <- function(cluster, ...) {
    metric_result("noise_points", sum(cluster == 0, na.rm = TRUE), "minimize", "internal")
  }

  metric_loglik <- function(cluster, obj = NULL, ...) {
    metric_from_attr(cluster, "loglik", "maximize", "model", obj = obj)
  }

  metric_withinerror <- function(cluster, obj = NULL, ...) {
    metric_from_attr(cluster, "withinerror", "minimize", "model", obj = obj)
  }

  metric_modularity <- function(cluster, obj = NULL, ...) {
    metric_from_attr(cluster, "modularity", "maximize", "model", obj = obj)
  }

  selector_best <- function(values, goal = c("maximize", "minimize")) {
    goal <- match.arg(goal)
    values <- as.numeric(values)
    idx <- which(!is.na(values))
    if (length(idx) == 0) return(NA_integer_)
    values_ok <- values[idx]
    best <- if (goal == "maximize") max(values_ok) else min(values_ok)
    idx[which(values_ok == best)[1]]
  }

  selector_elbow <- function(values, goal = c("minimize", "maximize")) {
    goal <- match.arg(goal)
    values <- as.numeric(values)
    idx <- which(!is.na(values))
    if (length(idx) <= 2) {
      return(selector_best(values, goal = goal))
    }

    curve <- values[idx]
    if (goal == "maximize") {
      curve <- -curve
    }

    myfit <- fit_curvature_max()
    res <- transform(myfit, curve)
    idx[res$x[1]]
  }

  obj <- dal_base()
  class(obj) <- append("cluutils", class(obj))
  obj$metric_result <- metric_result
  obj$metric_from_attr <- metric_from_attr
  obj$metric_wcss <- metric_wcss
  obj$metric_silhouette <- metric_silhouette
  obj$metric_davies_bouldin <- metric_davies_bouldin
  obj$metric_calinski_harabasz <- metric_calinski_harabasz
  obj$metric_entropy <- metric_entropy
  obj$metric_purity <- metric_purity
  obj$metric_rand_index <- metric_rand_index
  obj$metric_adjusted_rand_index <- metric_adjusted_rand_index
  obj$metric_noise_points <- metric_noise_points
  obj$metric_loglik <- metric_loglik
  obj$metric_withinerror <- metric_withinerror
  obj$metric_modularity <- metric_modularity
  obj$selector_best <- selector_best
  obj$selector_elbow <- selector_elbow
  return(obj)
}
