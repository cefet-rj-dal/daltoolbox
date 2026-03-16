#'@title Basic imputation
#'@description Impute missing values in mixed datasets using simple statistics.
#'@details Numeric columns are imputed with the mean or median. Factor, character,
#' logical, and ordered columns are imputed with the mode (most frequent observed value).
#' This class is intended as a low-complexity baseline for preprocessing workflows.
#' The default recommendation of median for numeric variables follows standard data
#' preprocessing guidance because it is less sensitive to outliers than the mean,
#' while mode imputation is the usual baseline for categorical attributes.
#'@param numeric_method imputation method for numeric columns: "median" or "mean"
#'@param cols optional vector of column names to impute (default: all supported columns)
#'@return returns an object of class `imputation_basic`
#'@references
#' Han, J., Kamber, M., Pei, J. (2011). Data Mining: Concepts and Techniques.
#'
#' Little, R. J. A., Rubin, D. B. (2019). Statistical Analysis with Missing Data.
#'@examples
#'data(iris)
#'iris_na <- iris
#'iris_na$Sepal.Length[c(2, 10, 25)] <- NA
#'iris_na$Species[c(3, 15)] <- NA
#'
#'imp <- imputation_basic()
#'imp <- fit(imp, iris_na)
#'iris_imp <- transform(imp, iris_na)
#'summary(iris_imp$Sepal.Length)
#'table(iris_imp$Species, useNA = "ifany")
#'@export
imputation_basic <- function(numeric_method = c("median", "mean"), cols = NULL) {
  obj <- dal_transform()
  obj$numeric_method <- match.arg(numeric_method)
  obj$cols <- cols
  class(obj) <- append("imputation_basic", class(obj))
  return(obj)
}

imputation_basic_mode <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) {
    stop("imputation_basic: cannot compute mode for data with only missing values.")
  }
  values <- unique(x)
  values[[which.max(tabulate(match(x, values)))]]
}

imputation_basic_kind <- function(x) {
  if (is.numeric(x)) {
    return("numeric")
  }
  if (is.ordered(x)) {
    return("ordered")
  }
  if (is.factor(x)) {
    return("factor")
  }
  if (is.character(x)) {
    return("character")
  }
  if (is.logical(x)) {
    return("logical")
  }
  stop("imputation_basic: unsupported column type for imputation.")
}

#'@exportS3Method fit imputation_basic
fit.imputation_basic <- function(obj, data, ...) {
  numeric_method <- obj$numeric_method

  if (is.vector(data) && !is.list(data)) {
    kind <- imputation_basic_kind(data)
    value <- if (kind == "numeric") {
      if (all(is.na(data))) {
        stop("imputation_basic: numeric vector contains only missing values.")
      }
      if (numeric_method == "median") stats::median(data, na.rm = TRUE) else mean(data, na.rm = TRUE)
    } else {
      imputation_basic_mode(data)
    }
    obj$values <- value
    obj$kinds <- kind
    obj$vector <- TRUE
    return(obj)
  }

  data <- adjust_data.frame(data)
  cols <- obj$cols
  if (is.null(cols)) {
    supported <- sapply(data, function(col) {
      is.numeric(col) || is.ordered(col) || is.factor(col) || is.character(col) || is.logical(col)
    })
    cols <- names(data)[supported]
  }

  if (length(cols) == 0) {
    stop("imputation_basic: no supported columns to impute.")
  }
  if (!all(cols %in% names(data))) {
    stop("imputation_basic: some columns in 'cols' are not in data.")
  }

  kinds <- vapply(data[, cols, drop = FALSE], imputation_basic_kind, character(1))
  values <- lapply(cols, function(col) {
    column <- data[[col]]
    kind <- kinds[[col]]
    if (kind == "numeric") {
      if (all(is.na(column))) {
        stop(paste0("imputation_basic: column '", col, "' contains only missing values."))
      }
      if (numeric_method == "median") stats::median(column, na.rm = TRUE) else mean(column, na.rm = TRUE)
    } else {
      imputation_basic_mode(column)
    }
  })
  names(values) <- cols

  obj$cols <- cols
  obj$kinds <- kinds
  obj$values <- values
  obj$vector <- FALSE
  return(obj)
}

#'@exportS3Method transform imputation_basic
transform.imputation_basic <- function(obj, data, ...) {
  if (isTRUE(obj$vector)) {
    data[is.na(data)] <- obj$values
    return(data)
  }

  data <- adjust_data.frame(data)
  cols <- obj$cols
  if (length(cols) == 0) {
    return(data)
  }

  for (col in cols) {
    if (!col %in% names(data)) {
      next
    }
    idx <- is.na(data[[col]])
    if (any(idx)) {
      data[[col]][idx] <- obj$values[[col]]
    }
  }
  return(data)
}
