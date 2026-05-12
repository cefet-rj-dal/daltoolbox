#'@title Class balancing (up/down sampling)
#'@description Balance class distribution using up-sampling or down-sampling.
#'@param attribute target class attribute name
#'@param method balancing method: "down" or "up"
#'@return returns an object of class `sample_balance`
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

#'@importFrom caret downSample
#'@importFrom caret upSample
#'@exportS3Method transform sample_balance
transform.sample_balance <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  attribute <- obj$attribute
  if (is.null(attribute) || !attribute %in% names(data)) {
    stop("sample_balance: 'attribute' must be a valid column name in data.")
  }
  x <- data[, setdiff(names(data), attribute), drop=FALSE]
  y <- data[[attribute]]
  if (obj$method == "down") {
    res <- caret::downSample(x = x, y = y)
  } else {
    res <- caret::upSample(x = x, y = y)
  }
  res[[attribute]] <- res$Class
  res$Class <- NULL
  return(res)
}
