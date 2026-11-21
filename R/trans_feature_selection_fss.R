#'@title Feature selection by forward stepwise search
#'@description Selects numeric predictors using forward stepwise subset search.
#'@details Adds one predictor at a time (greedy forward selection) and keeps the subset
#' with the highest adjusted R-squared, which penalizes adding predictors that do not pay
#' for their complexity. The target attribute must be numeric.
#'
#' This is an independent native implementation written from the subset-selection
#' literature (see references), using base `stats::lm` only, so the package carries no
#' external feature-selection dependency.
#'@param attribute target attribute name
#'@param features optional vector of feature names (default: all columns except `attribute`)
#'@return returns an object of class `feature_selection_fss`
#'@references
#' Hocking, R. R. (1976). The Analysis and Selection of Variables in Linear Regression. Biometrics, 32(1):1-49.
#' Miller, A. (2002). Subset Selection in Regression (2nd ed.). Chapman & Hall/CRC.
#'@examples
#'data(mtcars)
#'fs <- feature_selection_fss("mpg")
#'fs <- fit(fs, mtcars)
#'fs$selected
#'mtcars_fs <- transform(fs, mtcars)
#'names(mtcars_fs)
#'@export
feature_selection_fss <- function(attribute, features = NULL) {
  obj <- dal_transform()
  obj$attribute <- attribute
  obj$features <- features
  class(obj) <- append("feature_selection_fss", class(obj))
  return(obj)
}

# Adjusted R-squared of lm(attribute ~ cols) on data; -Inf for an empty/degenerate model.
fss_adj_r2 <- function(data, attribute, cols) {
  if (length(cols) == 0) return(-Inf)
  fml <- stats::as.formula(paste(attribute, "~", paste(cols, collapse = " + ")))
  val <- summary(stats::lm(fml, data = data))$adj.r.squared
  if (is.null(val) || !is.finite(val)) -Inf else val
}

#'@importFrom stats lm as.formula
#'@exportS3Method fit feature_selection_fss
fit.feature_selection_fss <- function(obj, data, ...) {
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

  # greedy forward selection: add the variable that most improves adjusted R-squared,
  # then keep the forward-path size with the highest adjusted R-squared.
  selected <- character(0)
  remaining <- numeric_features
  order_sel <- character(0)
  scores <- numeric(0)
  while (length(remaining) > 0) {
    cand_scores <- vapply(
      remaining,
      function(f) fss_adj_r2(data, attr, c(selected, f)),
      numeric(1)
    )
    best <- which.max(cand_scores)
    best_f <- remaining[best]
    selected <- c(selected, best_f)
    remaining <- setdiff(remaining, best_f)
    order_sel <- c(order_sel, best_f)
    scores <- c(scores, cand_scores[best])
  }
  best_size <- which.max(scores)
  selected <- order_sel[seq_len(best_size)]

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
