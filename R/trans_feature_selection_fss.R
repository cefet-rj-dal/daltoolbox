#'@title Feature selection by forward stepwise search
#'@description Selects numeric predictors using forward stepwise subset search.
#'@details Uses `leaps::regsubsets` and keeps the subset with the highest adjusted R-squared.
#' The target attribute must be numeric.
#'@param attribute target attribute name
#'@param features optional vector of feature names (default: all columns except `attribute`)
#'@return returns an object of class `feature_selection_fss`
#'@examples
#'if (requireNamespace("leaps", quietly = TRUE)) {
#'  data(Boston)
#'  fs <- feature_selection_fss("medv")
#'  fs <- fit(fs, Boston)
#'  fs$selected
#'  boston_fs <- transform(fs, Boston)
#'  names(boston_fs)
#'}
#'@export
feature_selection_fss <- function(attribute, features = NULL) {
  obj <- dal_transform()
  obj$attribute <- attribute
  obj$features <- features
  class(obj) <- append("feature_selection_fss", class(obj))
  return(obj)
}

#'@exportS3Method fit feature_selection_fss
fit.feature_selection_fss <- function(obj, data, ...) {
  if (!requireNamespace("leaps", quietly = TRUE)) {
    stop("feature_selection_fss requires the 'leaps' package. Install with install.packages('leaps').")
  }

  data <- adjust_data.frame(data)
  attr <- obj$attribute
  if (is.null(attr) || !attr %in% names(data)) {
    stop("feature_selection_fss: attribute not found in data.")
  }

  if (!is.numeric(data[[attr]])) {
    stop("feature_selection_fss: target attribute must be numeric.")
  }

  features <- obj$features
  if (is.null(features)) {
    features <- setdiff(names(data), attr)
  }
  features <- intersect(features, names(data))

  numeric_features <- features[vapply(data[features], is.numeric, logical(1))]
  obj$features <- numeric_features

  if (length(numeric_features) == 0) {
    obj$ranking <- data.frame(feature = character(0), score = numeric(0), stringsAsFactors = FALSE)
    obj$selected <- character(0)
    return(obj)
  }

  predictors <- as.matrix(data[, numeric_features, drop = FALSE])
  predictand <- data[[attr]]
  regfit <- leaps::regsubsets(
    predictors,
    predictand,
    nvmax = length(numeric_features),
    method = "forward"
  )
  regsum <- summary(regfit)
  best_size <- which.max(regsum$adjr2)
  coef_names <- names(stats::coef(regfit, best_size))
  selected <- setdiff(coef_names, "(Intercept)")

  ranking <- data.frame(
    feature = selected,
    score = seq_along(selected),
    stringsAsFactors = FALSE
  )

  obj$selected <- selected
  obj$ranking <- ranking
  return(obj)
}

#'@exportS3Method transform feature_selection_fss
transform.feature_selection_fss <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  if (is.null(obj$selected)) {
    stop("feature_selection_fss: call fit() before transform().")
  }
  keep <- c(obj$attribute, obj$selected)
  keep <- intersect(keep, names(data))
  data <- data[, keep, drop = FALSE]
  return(data)
}
