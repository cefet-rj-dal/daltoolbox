#'@title Feature selection by lasso
#'@description Selects predictors using L1-regularized regression.
#'@details Fits a lasso path with `glmnet` and keeps predictors with non-zero coefficients at `lambda.min`.
#'@param attribute target attribute name
#'@param features optional vector of feature names (default: all numeric columns except `attribute`)
#'@return returns an object of class `feature_selection_lasso`
#'@examples
#'if (requireNamespace("glmnet", quietly = TRUE)) {
#'  data(iris)
#'  fs <- feature_selection_lasso("Sepal.Length")
#'  fs <- fit(fs, iris)
#'  fs$selected
#'  iris_fs <- transform(fs, iris)
#'  names(iris_fs)
#'}
#'@export
feature_selection_lasso <- function(attribute, features = NULL) {
  obj <- dal_transform()
  obj$attribute <- attribute
  obj$features <- features
  class(obj) <- append("feature_selection_lasso", class(obj))
  return(obj)
}

#'@exportS3Method fit feature_selection_lasso
fit.feature_selection_lasso <- function(obj, data, ...) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("feature_selection_lasso requires the 'glmnet' package. Install with install.packages('glmnet').")
  }

  data <- adjust_data.frame(data)
  attr <- obj$attribute
  if (is.null(attr) || !attr %in% names(data)) {
    stop("feature_selection_lasso: attribute not found in data.")
  }

  if (!is.numeric(data[[attr]])) {
    data[[attr]] <- as.numeric(data[[attr]])
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

  x <- as.matrix(data[, numeric_features, drop = FALSE])
  y <- data[[attr]]
  cvfit <- glmnet::cv.glmnet(x, y, alpha = 1)
  coef_mat <- as.matrix(stats::coef(cvfit, s = "lambda.min"))
  nz <- coef_mat[, 1] != 0
  selected <- setdiff(rownames(coef_mat)[nz], "(Intercept)")

  ranking <- data.frame(
    feature = selected,
    score = abs(coef_mat[selected, 1]),
    stringsAsFactors = FALSE
  )
  if (nrow(ranking) > 1) {
    ranking <- ranking[order(ranking$score, decreasing = TRUE), , drop = FALSE]
  }

  obj$model <- cvfit
  obj$selected <- selected
  obj$ranking <- ranking
  return(obj)
}

#'@exportS3Method transform feature_selection_lasso
transform.feature_selection_lasso <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  if (is.null(obj$selected)) {
    stop("feature_selection_lasso: call fit() before transform().")
  }
  keep <- c(obj$attribute, obj$selected)
  keep <- intersect(keep, names(data))
  data <- data[, keep, drop = FALSE]
  return(data)
}