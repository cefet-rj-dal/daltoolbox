#'@title Feature selection by RELIEF
#'@description Rank and select features using a simplified RELIEF algorithm.
#'@details For each sampled instance, the algorithm compares nearest hit/miss neighbors and updates feature weights.
#'@param attribute target attribute name
#'@param features optional vector of feature names (default: all columns except `attribute`)
#'@param top optional number of top features to keep
#'@param cutoff optional minimum RELIEF weight to keep a feature
#'@param m number of sampled instances for RELIEF updates
#'@param seed random seed for sampling
#'@return returns an object of class `feature_selection_relief`
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
#'@export
feature_selection_relief <- function(attribute, features = NULL, top = NULL, cutoff = NULL, m = 50, seed = 1) {
  obj <- dal_transform()
  obj$attribute <- attribute
  obj$features <- features
  obj$top <- top
  obj$cutoff <- cutoff
  obj$m <- m
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
  set.seed(obj$seed)
  idxs <- sample(seq_len(n), size = m)
  w <- rep(0, ncol(X))

  for (i in idxs) {
    xi <- X[i, , drop = FALSE]
    yi <- y[i]
    d <- rowSums((X - matrix(xi, nrow = n, ncol = ncol(X), byrow = TRUE))^2)

    same <- which(y == yi & seq_len(n) != i)
    diff <- which(y != yi)
    if (length(same) == 0 || length(diff) == 0) {
      next
    }
    nh <- same[which.min(d[same])]
    nm <- diff[which.min(d[diff])]
    w <- w - abs(X[i, ] - X[nh, ]) + abs(X[i, ] - X[nm, ])
  }

  w <- w / m
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
