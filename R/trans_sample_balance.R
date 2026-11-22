#'@title Class balancing (up/down sampling)
#'@description Balance class distribution using up-sampling or down-sampling.
#'@details Down-sampling reduces every class to the minority count (without replacement);
#' up-sampling grows every class to the majority count (sampling with replacement). Native
#' implementation (no external dependency).
#'@param attribute target class attribute name
#'@param method balancing method: "down" or "up"
#'@return returns an object of class `sample_balance`
#'@references
#' He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. IEEE Transactions on Knowledge and Data Engineering, 21(9):1263-1284.
#'@examples
#'data(iris)
#'iris_imb <- iris[iris$Species != "setosa", ]
#'sb <- sample_balance("Species", method = "down")
#'iris_bal <- transform(sb, iris_imb)
#'table(iris_bal$Species)
#'@export
sample_balance <- function(attribute, method = c("down", "up")) {
  obj <- dal_transform()
  obj$attribute <- attribute
  obj$method <- match.arg(method)
  class(obj) <- append("sample_balance", class(obj))
  return(obj)
}

#'@exportS3Method transform sample_balance
transform.sample_balance <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  attribute <- obj$attribute
  if (is.null(attribute) || !attribute %in% names(data)) {
    stop("sample_balance: 'attribute' must be a valid column name in data.")
  }
  y <- as.character(data[[attribute]])
  counts <- table(y)
  counts <- counts[counts > 0]
  target <- if (obj$method == "down") min(counts) else max(counts)
  parts <- list()
  for (cls in names(counts)) {
    idx <- which(y == cls)
    replace <- length(idx) < target
    sel <- sample(idx, size = target, replace = replace)
    parts[[length(parts) + 1]] <- data[sel, , drop = FALSE]
  }
  result <- do.call(rbind, parts)
  rownames(result) <- NULL
  return(result)
}
