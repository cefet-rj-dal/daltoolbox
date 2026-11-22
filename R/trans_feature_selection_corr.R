#'@title Feature selection by correlation
#'@description Remove highly correlated numeric features based on a correlation cutoff.
#'@details For each pair of numeric features whose absolute correlation exceeds `cutoff`,
#' the one with the larger mean absolute correlation to the remaining features is removed,
#' reducing redundancy among predictors.
#'@references
#' Hall, M. A. (1999). Correlation-based Feature Selection for Machine Learning. PhD thesis, University of Waikato.
#' Kuhn, M. (2008). Building Predictive Models in R Using the caret Package. Journal of Statistical Software, 28(5):1-26.
#'@param cutoff correlation cutoff in \[0, 1\] above which one feature is removed
#'@param features optional vector of feature names to consider (default: all numeric columns)
#'@param keep optional vector of columns that should always be kept in `transform()`
#'@return returns an object of class `feature_selection_corr`
#'@examples
#'data(iris)
#'fs <- feature_selection_corr(cutoff = 0.9)
#'fs <- fit(fs, iris)
#'iris_fs <- transform(fs, iris)
#'fs$selected
#'names(iris_fs)
#'@export
feature_selection_corr <- function(cutoff = 0.9, features = NULL, keep = NULL) {
  obj <- dal_transform()
  obj$cutoff <- cutoff
  obj$features <- features
  obj$keep <- keep
  class(obj) <- append("feature_selection_corr", class(obj))
  return(obj)
}

# For each highly correlated pair, flag the member with the larger mean absolute
# correlation to the rest; returns the column indices to remove.
fs_find_correlation <- function(cor_mat, cutoff) {
  m <- abs(cor_mat)
  cols <- colnames(m)
  p <- length(cols)
  mean_abs <- colMeans(m)
  remove <- character(0)
  for (i in seq_len(p)) {
    for (j in seq_len(p)) {
      if (j <= i) next
      a <- cols[i]
      b <- cols[j]
      if (a %in% remove || b %in% remove) next
      if (m[a, b] > cutoff) {
        remove <- c(remove, if (mean_abs[a] >= mean_abs[b]) a else b)
      }
    }
  }
  which(cols %in% remove)
}

#'@exportS3Method fit feature_selection_corr
fit.feature_selection_corr <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  nums <- names(data)[sapply(data, is.numeric)]
  features <- obj$features
  if (!is.null(features)) {
    nums <- intersect(nums, features)
  }
  obj$features <- nums

  if (length(nums) <= 1) {
    obj$remove <- character(0)
    obj$selected <- nums
    obj$ranking <- data.frame(
      feature = nums,
      score = rep(0, length(nums)),
      stringsAsFactors = FALSE
    )
    return(obj)
  }

  cor_mat <- stats::cor(data[, nums, drop=FALSE], use = "pairwise.complete.obs")
  remove_idx <- fs_find_correlation(cor_mat, cutoff = obj$cutoff)
  obj$remove <- nums[remove_idx]
  obj$selected <- setdiff(nums, obj$remove)
  score <- sapply(obj$selected, function(f) {
    max(abs(cor_mat[f, setdiff(nums, f)]), na.rm = TRUE)
  })
  obj$ranking <- data.frame(
    feature = names(sort(score, decreasing = FALSE)),
    score = as.numeric(sort(score, decreasing = FALSE)),
    stringsAsFactors = FALSE
  )
  return(obj)
}

#'@exportS3Method transform feature_selection_corr
transform.feature_selection_corr <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  if (is.null(obj$selected)) {
    stop("feature_selection_corr: call fit() before transform().")
  }
  keep <- obj$keep
  if (is.null(keep)) {
    keep <- setdiff(names(data), obj$features)
  } else {
    keep <- intersect(keep, names(data))
  }
  cols <- unique(c(keep, obj$selected))
  data <- data[, cols, drop = FALSE]
  return(data)
}
