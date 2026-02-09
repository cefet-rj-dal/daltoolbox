#'@title Simple imputation
#'@description Impute missing values in numeric columns using the mean or median.
#'@param method imputation method: "median" or "mean"
#'@param cols optional vector of column names to impute (default: all numeric columns)
#'@return returns an object of class `imputation_simple`
#'@examples
#'data(iris)
#'iris_na <- iris
#'iris_na$Sepal.Length[c(2, 10, 25)] <- NA
#'
#'imp <- imputation_simple(method = "median")
#'imp <- fit(imp, iris_na)
#'iris_imp <- transform(imp, iris_na)
#'summary(iris_imp$Sepal.Length)
#'@export
imputation_simple <- function(method = c("median", "mean"), cols = NULL) {
  obj <- dal_transform()
  obj$method <- match.arg(method)
  obj$cols <- cols
  class(obj) <- append("imputation_simple", class(obj))
  return(obj)
}

#'@exportS3Method fit imputation_simple
fit.imputation_simple <- function(obj, data, ...) {
  method <- obj$method
  if (is.vector(data) && !is.list(data)) {
    if (!is.numeric(data)) {
      stop("imputation_simple: vector input must be numeric.")
    }
    value <- if (method == "median") stats::median(data, na.rm = TRUE) else mean(data, na.rm = TRUE)
    obj$values <- value
    obj$vector <- TRUE
    return(obj)
  }

  data <- adjust_data.frame(data)
  cols <- obj$cols
  if (is.null(cols)) {
    cols <- names(data)[sapply(data, is.numeric)]
  }
  if (length(cols) == 0) {
    stop("imputation_simple: no numeric columns to impute.")
  }
  if (!all(cols %in% names(data))) {
    stop("imputation_simple: some columns in 'cols' are not in data.")
  }
  if (!all(sapply(data[, cols, drop=FALSE], is.numeric))) {
    stop("imputation_simple: all 'cols' must be numeric.")
  }

  vals <- vapply(cols, function(col) {
    if (method == "median") stats::median(data[[col]], na.rm = TRUE) else mean(data[[col]], na.rm = TRUE)
  }, numeric(1))
  obj$cols <- cols
  obj$values <- vals
  obj$vector <- FALSE
  return(obj)
}

#'@exportS3Method transform imputation_simple
transform.imputation_simple <- function(obj, data, ...) {
  if (isTRUE(obj$vector)) {
    data[is.na(data)] <- obj$values
    return(data)
  }

  data <- adjust_data.frame(data)
  cols <- obj$cols
  if (length(cols) == 0) {
    return(data)
  }
  vals <- obj$values
  for (col in cols) {
    data[[col]][is.na(data[[col]])] <- vals[[col]]
  }
  return(data)
}
