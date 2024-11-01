#'@title TSRegSW
#'@description Time Series Regression from Sliding Windows.
#'Ancestral class for Machine Learning Implementation.
#'@param preprocess normalization
#'@param input_size input size for machine learning model
#'@return returns a `ts_regsw` object
#'@examples
#'#See ?ts_elm for an example using Extreme Learning Machine
#'@export
ts_regsw <- function(preprocess=NA, input_size=NA) {
  obj <- ts_reg()
  obj$ts_as_matrix <- function(data, input_size) {
    result <- data[,(ncol(data)-input_size+1):ncol(data)]
    return(result)
  }
  obj$preprocess <- preprocess
  obj$input_size <- input_size

  class(obj) <- append("ts_regsw", class(obj))
  return(obj)
}

#'@export
fit.ts_regsw <- function(obj, x, y, ...) {
  obj$preprocess <- fit(obj$preprocess, x)

  x <- transform(obj$preprocess, x)

  y <- transform(obj$preprocess, x, y)

  obj <- do_fit(obj, obj$ts_as_matrix(x, obj$input_size), y)

  return(obj)
}

#'@export
predict.ts_regsw <- function(object, x, steps_ahead=1, ...) {
  if (steps_ahead == 1) {
    x <- transform(object$preprocess, x)
    data <- object$ts_as_matrix(x, object$input_size)
    y <- do_predict(object, data)
    y <- inverse_transform(object$preprocess, x, y)
    return(as.vector(y))
  }
  else {
    if (nrow(x) > 1)
      stop("In steps ahead, x should have a single row")
    prediction <- NULL
    cnames <- colnames(x)
    x <- x[1,]
    for (i in 1:steps_ahead) {
      colnames(x) <- cnames
      x <- transform(object$preprocess, x)
      y <- do_predict(object, object$ts_as_matrix(x, object$input_size))
      x <- adjust_ts_data(inverse_transform(object$preprocess, x))
      y <- inverse_transform(object$preprocess, x, y)
      for (j in 1:(ncol(x)-1)) {
        x[1, j] <- x[1, j+1]
      }
      x[1, ncol(x)] <- y
      prediction <- c(prediction, y)
    }
    return(as.vector(prediction))
  }
  return(prediction)
}

#'@export
#'@importFrom stats predict
do_predict.ts_regsw <- function(obj, x) {
  prediction <- stats::predict(obj$model, x)
  return(prediction)
}


