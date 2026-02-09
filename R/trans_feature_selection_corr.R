#'@title Feature selection by correlation
#'@description Remove highly correlated numeric features based on a correlation cutoff.
#'@details Uses `caret::findCorrelation` on the correlation matrix computed from numeric columns.
#'@param cutoff correlation cutoff in \[0, 1\] above which one feature is removed
#'@return returns an object of class `feature_selection_corr`
#'@examples
#'data(iris)
#'fs <- feature_selection_corr(cutoff = 0.9)
#'fs <- fit(fs, iris)
#'iris_fs <- transform(fs, iris)
#'names(iris_fs)
#'@export
feature_selection_corr <- function(cutoff = 0.9) {
  obj <- dal_transform()
  obj$cutoff <- cutoff
  class(obj) <- append("feature_selection_corr", class(obj))
  return(obj)
}

#'@importFrom caret findCorrelation
#'@exportS3Method fit feature_selection_corr
fit.feature_selection_corr <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  nums <- names(data)[sapply(data, is.numeric)]
  if (length(nums) <= 1) {
    obj$remove <- character(0)
    return(obj)
  }

  cor_mat <- stats::cor(data[, nums, drop=FALSE], use = "pairwise.complete.obs")
  remove_idx <- caret::findCorrelation(cor_mat, cutoff = obj$cutoff)
  obj$remove <- nums[remove_idx]
  return(obj)
}

#'@exportS3Method transform feature_selection_corr
transform.feature_selection_corr <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  remove <- obj$remove
  if (!is.null(remove) && length(remove) > 0) {
    data[, remove] <- NULL
  }
  return(data)
}
