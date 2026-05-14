#'@title Predictive imputation base
#'@description Base class for supervised imputers that learn one target column from a set of source columns.
#'@details The target column is the attribute to be imputed. The source columns are the predictors used
#' to estimate missing target values. If `sources = NULL`, all supported columns except the target are used.
#' Missing values in source columns can be pre-imputed by a simpler method before fitting the predictive model.
#'@param target target column to impute
#'@param sources optional vector of predictor column names
#'@param method initial imputation method for numeric source columns: "median" or "mean"
#'@return returns an object of class `imputation_predictive`
#'@examples
#'data(iris)
#'imp <- imputation_predictive("Sepal.Length", sources = c("Sepal.Width", "Petal.Length", "Petal.Width", "Species"))
#'class(imp)
#'@export
imputation_predictive <- function(target, sources = NULL, method = c("median", "mean")) {
  obj <- dal_transform()
  obj$target <- target
  obj$sources <- sources
  obj$method <- match.arg(method)
  class(obj) <- append("imputation_predictive", class(obj))
  return(obj)
}

imputation_predictive_resolve <- function(obj, data) {
  data <- adjust_data.frame(data)
  target <- obj$target
  if (is.null(target) || !target %in% names(data)) {
    stop("imputation_predictive: 'target' must be a valid column name in data.", call. = FALSE)
  }

  supported <- sapply(data, function(col) {
    is.numeric(col) || is.ordered(col) || is.factor(col) || is.character(col) || is.logical(col)
  })
  supported_cols <- names(data)[supported]
  if (!target %in% supported_cols) {
    stop("imputation_predictive: target column type is not supported.", call. = FALSE)
  }

  sources <- obj$sources
  if (is.null(sources)) {
    sources <- setdiff(supported_cols, target)
  }
  if (!all(sources %in% names(data))) {
    stop("imputation_predictive: some columns in 'sources' are not in data.", call. = FALSE)
  }
  sources <- intersect(sources, supported_cols)
  if (length(sources) == 0) {
    stop("imputation_predictive: at least one supported source column is required.", call. = FALSE)
  }

  target_kind <- imputation_simple_kind(data[[target]])
  source_kinds <- vapply(data[, sources, drop = FALSE], imputation_simple_kind, character(1))
  all_kinds <- c(setNames(target_kind, target), source_kinds)
  levels_map <- lapply(c(target, sources), function(col) {
    imputation_tree_get_levels(data[[col]], all_kinds[[col]])
  })
  names(levels_map) <- c(target, sources)

  if (all(is.na(data[[target]]))) {
    stop(paste0("imputation_predictive: target column '", target, "' contains only missing values."), call. = FALSE)
  }
  for (col in sources) {
    if (all(is.na(data[[col]]))) {
      stop(paste0("imputation_predictive: source column '", col, "' contains only missing values."), call. = FALSE)
    }
  }

  list(
    data = data,
    target = target,
    sources = sources,
    target_kind = target_kind,
    source_kinds = source_kinds,
    levels_map = levels_map
  )
}
