#'@title Tree-based predictive imputation
#'@description Impute one target column from a set of source columns using a decision tree.
#'@details The method fits a tree with the observed values of the target column and uses the
#' source columns as predictors. If source columns contain missing values, they are first
#' completed with `imputation_simple()` so the tree can be trained and applied. The learned
#' model imputes only the target column; source columns are preserved in the returned data.
#'@param target target column to impute
#'@param sources optional vector of predictor column names (default: all supported columns except `target`)
#'@param method initial imputation method for numeric source columns: "median" or "mean"
#'@return returns an object of class `imputation_tree`
#'@references
#' Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984).
#' Classification and Regression Trees. Wadsworth.
#'
#' van Buuren, S., Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by
#' Chained Equations in R. Journal of Statistical Software, 45(3), 1-67.
#'@examples
#'data(iris)
#'iris_na <- iris
#'iris_na$Sepal.Length[c(2, 10, 25)] <- NA
#'
#'imp <- imputation_tree("Sepal.Length")
#'imp <- fit(imp, iris_na)
#'iris_imp <- transform(imp, iris_na)
#'summary(iris_imp$Sepal.Length)
#'sum(is.na(iris_imp$Sepal.Length))
#'@export
imputation_tree <- function(target, sources = NULL, method = c("median", "mean")) {
  obj <- imputation_predictive(target = target, sources = sources, method = method)
  class(obj) <- append("imputation_tree", class(obj))
  return(obj)
}

imputation_tree_get_levels <- function(x, kind) {
  if (kind == "numeric") {
    return(NULL)
  }
  if (kind == "factor" || kind == "ordered") {
    return(levels(x))
  }
  if (kind == "logical") {
    return(c("FALSE", "TRUE"))
  }
  unique(as.character(x[!is.na(x)]))
}

imputation_tree_prepare_column <- function(x, kind, levels) {
  if (kind == "numeric") {
    return(as.numeric(x))
  }

  values <- as.character(x)
  prepared <- factor(values, levels = levels)
  invalid <- !is.na(x) & is.na(prepared)
  if (any(invalid)) {
    stop("imputation_tree: transform data contains unseen categorical levels.")
  }
  return(prepared)
}

imputation_tree_restore_column <- function(x, kind, levels) {
  if (kind == "numeric") {
    return(as.numeric(x))
  }

  values <- as.character(x)
  if (kind == "factor") {
    return(factor(values, levels = levels))
  }
  if (kind == "ordered") {
    return(ordered(values, levels = levels))
  }
  if (kind == "logical") {
    return(as.logical(values))
  }
  return(values)
}

imputation_tree_prepare_frame <- function(data, columns, kinds, levels_map) {
  frame <- data[, columns, drop = FALSE]
  for (col in columns) {
    frame[[col]] <- imputation_tree_prepare_column(frame[[col]], kinds[[col]], levels_map[[col]])
  }
  return(frame)
}

#'@exportS3Method fit imputation_tree
fit.imputation_tree <- function(obj, data, ...) {
  resolved <- imputation_predictive_resolve(obj, data)
  data <- resolved$data
  target <- resolved$target
  sources <- resolved$sources
  target_kind <- resolved$target_kind
  source_kinds <- resolved$source_kinds
  levels_map <- resolved$levels_map

  initial_model <- imputation_simple(method = obj$method, cols = sources)
  initial_model <- fit(initial_model, data)
  work <- transform(initial_model, data)

  observed <- !is.na(data[[target]])
  if (!any(observed)) {
    stop("imputation_tree: target has no observed values for model fitting.", call. = FALSE)
  }

  train_kinds <- c(stats::setNames(target_kind, target), source_kinds)
  train_frame <- work[observed, c(target, sources), drop = FALSE]
  train_frame <- imputation_tree_prepare_frame(train_frame, c(target, sources), train_kinds, levels_map)

  obj$target <- target
  obj$sources <- sources
  obj$target_kind <- target_kind
  obj$source_kinds <- source_kinds
  obj$levels <- levels_map
  obj$initial_model <- initial_model
  obj$model <- tree::tree(stats::formula(paste(target, "~ .")), train_frame)
  return(obj)
}

#'@exportS3Method transform imputation_tree
transform.imputation_tree <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  target <- obj$target
  sources <- obj$sources

  if (!target %in% names(data)) {
    stop("imputation_tree: target column not found in transform data.", call. = FALSE)
  }

  missing_sources <- setdiff(sources, names(data))
  if (length(missing_sources) > 0) {
    stop(paste0(
      "imputation_tree: missing source columns in transform data: ",
      paste(missing_sources, collapse = ", ")
    ), call. = FALSE)
  }

  missing_idx <- is.na(data[[target]])
  if (!any(missing_idx)) {
    return(data)
  }

  data_work <- transform(obj$initial_model, data)
  pred_frame <- data_work[missing_idx, sources, drop = FALSE]
  pred_frame <- imputation_tree_prepare_frame(pred_frame, sources, obj$source_kinds, obj$levels)
  prediction <- if (obj$target_kind == "numeric") {
    predict(obj$model, pred_frame, type = "vector")
  } else {
    predict(obj$model, pred_frame, type = "class")
  }
  prediction <- imputation_tree_restore_column(prediction, obj$target_kind, obj$levels[[target]])
  data[[target]][missing_idx] <- prediction
  return(data)
}
