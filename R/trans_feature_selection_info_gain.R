#'@title Feature selection by information gain
#'@description Rank and select features using information gain with optional discretization.
#'@details Numeric predictors are discretized by quantile bins before computing entropy-based information gain.
#'@param attribute target attribute name
#'@param features optional vector of feature names (default: all columns except `attribute`)
#'@param top optional number of top features to keep
#'@param cutoff minimum information gain to keep a feature (default: 0)
#'@param bins number of quantile bins for numeric features
#'@return returns an object of class `feature_selection_info_gain`
#'@examples
#'data(iris)
#'fg <- feature_generation(
#'  IsVersicolor = ifelse(Species == "versicolor", "versicolor", "not_versicolor")
#')
#'iris_bin <- transform(fg, iris)
#'iris_bin$IsVersicolor <- factor(iris_bin$IsVersicolor)
#'fs <- feature_selection_info_gain("IsVersicolor", top = 2)
#'fs <- fit(fs, iris_bin)
#'fs$selected
#'iris_fs <- transform(fs, iris_bin)
#'names(iris_fs)
#'@export
feature_selection_info_gain <- function(attribute, features = NULL, top = NULL, cutoff = 0, bins = 3) {
  obj <- dal_transform()
  obj$attribute <- attribute
  obj$features <- features
  obj$top <- top
  obj$cutoff <- cutoff
  obj$bins <- bins
  class(obj) <- append("feature_selection_info_gain", class(obj))
  return(obj)
}

#'@exportS3Method fit feature_selection_info_gain
fit.feature_selection_info_gain <- function(obj, data, ...) {
  entropy <- function(y) {
    p <- prop.table(table(y))
    -sum(p * log2(p))
  }
  make_bins <- function(x, bins) {
    q <- stats::quantile(x, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE)
    q <- unique(q)
    if (length(q) < 2) {
      return(factor(rep("all", length(x))))
    }
    cut(x, breaks = q, include.lowest = TRUE, labels = FALSE)
  }
  info_gain <- function(x, y, bins) {
    if (is.numeric(x)) {
      x <- make_bins(x, bins = bins)
    } else {
      x <- as.factor(x)
    }
    total <- entropy(y)
    cond <- 0
    lvls <- levels(as.factor(x))
    for (lvl in lvls) {
      idx <- which(as.character(x) == lvl)
      if (length(idx) > 0) {
        cond <- cond + (length(idx) / length(y)) * entropy(y[idx])
      }
    }
    total - cond
  }

  data <- adjust_data.frame(data)
  attr <- obj$attribute
  if (is.null(attr) || !attr %in% names(data)) {
    stop("feature_selection_info_gain: attribute not found in data.")
  }
  y <- as.factor(data[[attr]])

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

  scores <- sapply(features, function(f) info_gain(data[[f]], y, obj$bins))
  scores[!is.finite(scores)] <- 0
  ord <- order(scores, decreasing = TRUE)
  ranking <- data.frame(
    feature = features[ord],
    score = as.numeric(scores[ord]),
    stringsAsFactors = FALSE
  )

  selected <- ranking$feature[ranking$score >= obj$cutoff]
  if (!is.null(obj$top)) {
    selected <- head(selected, obj$top)
  }

  obj$ranking <- ranking
  obj$selected <- selected
  return(obj)
}

#'@exportS3Method transform feature_selection_info_gain
transform.feature_selection_info_gain <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  if (is.null(obj$selected)) {
    stop("feature_selection_info_gain: call fit() before transform().")
  }
  keep <- c(obj$attribute, obj$selected)
  keep <- intersect(keep, names(data))
  data <- data[, keep, drop = FALSE]
  return(data)
}
