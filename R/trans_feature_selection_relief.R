#'@title Feature selection by ReliefF
#'@description Rank and select features using the ReliefF weighting scheme.
#'@details For each of `m` sampled instances the algorithm finds its `k` nearest neighbors
#' of the same class (nearest hits) and its `k` nearest neighbors of each other class
#' (nearest misses) in the normalized feature space, and updates every feature's weight by
#' subtracting its average hit difference and adding its average miss difference, the latter
#' weighted by each other class's prior probability `P(C) / (1 - P(class(R)))`. Averaging
#' over `k` neighbors (ReliefF, Kononenko 1994) makes the estimate robust to noise, and the
#' class-prior weighting handles multi-class targets.
#'
#' This is a native implementation inspired by the original Relief and evolved to full
#' ReliefF (see references), carrying no external feature-selection dependency.
#'@param attribute target attribute name
#'@param features optional vector of feature names (default: all columns except `attribute`)
#'@param top optional number of top features to keep
#'@param cutoff optional minimum ReliefF weight to keep a feature
#'@param m number of sampled instances for ReliefF updates
#'@param k number of nearest hits/misses per class averaged in each update (default 10)
#'@param seed optional seed for the instance sampling (reproducibility)
#'@return returns an object of class `feature_selection_relief`
#'@references
#' Kira, K. and Rendell, L. A. (1992). A Practical Approach to Feature Selection. In Proceedings of the Ninth International Workshop on Machine Learning, 249-256.
#' Kononenko, I. (1994). Estimating Attributes: Analysis and Extensions of RELIEF. In Proceedings of the European Conference on Machine Learning, 171-182.
#'@examples
#'data(iris)
#'fg <- feature_generation(
#'  IsVersicolor = ifelse(Species == "versicolor", "versicolor", "not_versicolor")
#')
#'iris_bin <- transform(fg, iris)
#'iris_bin$IsVersicolor <- factor(iris_bin$IsVersicolor)
#'fs <- feature_selection_relief("IsVersicolor", top = 2, m = 50)
#'fs <- fit(fs, iris_bin)
#'fs$selected
#'transform(fs, iris_bin) |> names()
#'@importFrom utils head
#'@export
feature_selection_relief <- function(attribute, features = NULL, top = NULL, cutoff = NULL, m = 50, k = 10, seed = NULL) {
  obj <- dal_transform()
  obj$attribute <- attribute
  obj$features <- features
  obj$top <- top
  obj$cutoff <- cutoff
  obj$m <- m
  obj$k <- k
  obj$seed <- seed
  class(obj) <- append("feature_selection_relief", class(obj))
  return(obj)
}

#'@exportS3Method fit feature_selection_relief
fit.feature_selection_relief <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  attr <- obj$attribute
  if (is.null(attr) || !attr %in% names(data)) {
    stop("feature_selection_relief: attribute not found in data.")
  }
  y <- as.factor(data[[attr]])
  if (length(levels(y)) < 2) {
    stop("feature_selection_relief: target must have at least two classes.")
  }

  features <- obj$features
  if (is.null(features)) {
    features <- setdiff(names(data), attr)
  }
  features <- intersect(features, names(data))
  obj$features <- features

  if (length(features) == 0) {
    obj$ranking <- data.frame(feature = character(0), score = numeric(0), stringsAsFactors = FALSE)
    obj$selected <- character(0)
    return(obj)
  }

  X <- data[, features, drop = FALSE]
  X <- as.data.frame(lapply(X, function(col) {
    if (is.numeric(col)) {
      return(col)
    }
    as.numeric(as.factor(col))
  }))
  X <- as.matrix(X)

  mins <- apply(X, 2, min, na.rm = TRUE)
  maxs <- apply(X, 2, max, na.rm = TRUE)
  den <- ifelse((maxs - mins) == 0, 1, (maxs - mins))
  X <- sweep(sweep(X, 2, mins, "-"), 2, den, "/")

  n <- nrow(X)
  m <- min(obj$m, n)
  k <- if (is.null(obj$k)) 10 else obj$k
  if (!is.null(obj$seed)) set.seed(obj$seed)
  idxs <- sample(seq_len(n), size = m)
  w <- rep(0, ncol(X))

  classes <- levels(y)
  priors <- as.numeric(table(y)[classes]) / n
  names(priors) <- classes
  m_eff <- 0

  for (i in idxs) {
    xi <- X[i, ]
    yi <- as.character(y[i])
    # city-block distance on normalized features
    d <- rowSums(abs(X - matrix(xi, nrow = n, ncol = ncol(X), byrow = TRUE)))

    same <- which(y == yi & seq_len(n) != i)
    if (length(same) == 0 || priors[yi] >= 1) {
      next
    }
    # k nearest hits (same class)
    kk <- min(k, length(same))
    hits <- same[order(d[same])[seq_len(kk)]]
    hit_diff <- colSums(abs(sweep(X[hits, , drop = FALSE], 2, xi, "-"))) / kk
    w <- w - hit_diff
    # k nearest misses per other class, weighted by prior probability
    for (c in classes) {
      if (c == yi) next
      members <- which(y == c)
      if (length(members) == 0) next
      kk2 <- min(k, length(members))
      misses <- members[order(d[members])[seq_len(kk2)]]
      miss_diff <- colSums(abs(sweep(X[misses, , drop = FALSE], 2, xi, "-"))) / kk2
      w <- w + (priors[c] / (1 - priors[yi])) * miss_diff
    }
    m_eff <- m_eff + 1
  }

  if (m_eff > 0) w <- w / m_eff
  ord <- order(w, decreasing = TRUE)
  ranking <- data.frame(
    feature = features[ord],
    score = as.numeric(w[ord]),
    stringsAsFactors = FALSE
  )

  selected <- ranking$feature
  if (!is.null(obj$cutoff)) {
    selected <- ranking$feature[ranking$score >= obj$cutoff]
  }
  if (!is.null(obj$top)) {
    selected <- head(selected, obj$top)
  }

  obj$ranking <- ranking
  obj$selected <- selected
  return(obj)
}

#'@exportS3Method transform feature_selection_relief
transform.feature_selection_relief <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  if (is.null(obj$selected)) {
    stop("feature_selection_relief: call fit() before transform().")
  }
  keep <- c(obj$attribute, obj$selected)
  keep <- intersect(keep, names(data))
  data <- data[, keep, drop = FALSE]
  return(data)
}
