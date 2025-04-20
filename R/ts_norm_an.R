#'@title Time Series Adaptive Normalization
#'@description Transform data to a common scale while taking into account the
#' changes in the statistical properties of the data over time.
#'@param remove_outliers logical: if TRUE (default) outliers will be removed.
#'@param nw integer: window size.
#'@return returns a `ts_norm_an` object.
#'@examples
#'# time series to normalize
#'data(sin_data)
#'
#'# convert to sliding windows
#'ts <- ts_data(sin_data$y, 10)
#'ts_head(ts, 3)
#'summary(ts[,10])
#'
#'# normalization
#'preproc <- ts_norm_an()
#'preproc <- fit(preproc, ts)
#'tst <- transform(preproc, ts)
#'ts_head(tst, 3)
#'summary(tst[,10])
#'@export
ts_norm_an <- function(remove_outliers = TRUE, nw = 0) {
  obj <- dal_transform()
  obj$ma <- function(obj, data, func) {
    if (obj$nw != 0) {
      cols <- ncol(data) - ((obj$nw-1):0)
      data <- data[,cols]

    }
    an <- apply(data, 1, func, na.rm=TRUE)
  }
  obj$remove_outliers <- remove_outliers

  obj$an_mean <- mean
  obj$nw <- nw
  class(obj) <- append("ts_norm_an", class(obj))
  return(obj)
}

#'@export
fit.ts_norm_an <- function(obj, data, ...) {
  input <- data[,1:(ncol(data)-1)]
  an <- obj$ma(obj, input, obj$an_mean)
  data <- data - an #

  if (obj$remove_outliers) {
    out <- outliers()
    out <- fit(out, data)
    data <- transform(out, data)
  }

  obj$gmin <- min(data)
  obj$gmax <- max(data)

  return(obj)
}

#'@export
transform.ts_norm_an <- function(obj, data, x=NULL, ...) {
  if (!is.null(x)) {
    an <- attr(data, "an")
    x <- x - an #
    x <- (x - obj$gmin) / (obj$gmax-obj$gmin)
    return(x)
  }
  else {
    an <- obj$ma(obj, data, obj$an_mean)
    data <- data - an #
    data <- (data - obj$gmin) / (obj$gmax-obj$gmin)
    attr(data, "an") <- an
    return (data)
  }
}

#'@export
inverse_transform.ts_norm_an <- function(obj, data, x=NULL, ...) {
  an <- attr(data, "an")
  if (!is.null(x)) {
    x <- x * (obj$gmax-obj$gmin) + obj$gmin
    x <- x + an #
    return(x)
  }
  else {
    data <- data * (obj$gmax-obj$gmin) + obj$gmin
    data <- data + an #
    attr(data, "an") <- an
    return (data)
  }
}
