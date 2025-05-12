#'@title Time Series Sliding Window Min-Max
#'@description The ts_norm_swminmax function creates an object for normalizing a time series based on the "sliding window min-max scaling" method
#'@param outliers Indicate outliers transformation class. NULL can avoid outliers removal.
#'@return returns a `ts_norm_swminmax` object.
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
#'preproc <- ts_norm_swminmax()
#'preproc <- fit(preproc, ts)
#'tst <- transform(preproc, ts)
#'ts_head(tst, 3)
#'summary(tst[,10])
#'@export
ts_norm_swminmax <- function(outliers = outliers_boxplot()) {
  obj <- dal_transform()
  obj$outliers <- outliers
  class(obj) <- append("ts_norm_swminmax", class(obj))
  return(obj)
}

#'@exportS3Method fit ts_norm_swminmax
fit.ts_norm_swminmax <- function(obj, data, ...) {
  if (!is.null(obj$outliers)) {
    out <- obj$outliers
    out <- fit(out, data)
    data <- transform(out, data)
  }
  return(obj)
}

#'@exportS3Method transform ts_norm_swminmax
transform.ts_norm_swminmax <- function(obj, data, x=NULL, ...) {
  if (!is.null(x)) {
    i_min <- attr(data, "i_min")
    i_max <- attr(data, "i_max")
    x <- (x-i_min)/(i_max-i_min)
    return(x)
  }
  else {
    i_min <- apply(data, 1, min)
    i_max <- apply(data, 1, max)
    data <- (data-i_min)/(i_max-i_min)
    attr(data, "i_min") <- i_min
    attr(data, "i_max") <- i_max
    return(data)
  }
}

#'@exportS3Method inverse_transform ts_norm_swminmax
inverse_transform.ts_norm_swminmax <- function(obj, data, x=NULL, ...) {
  i_min <- attr(data, "i_min")
  i_max <- attr(data, "i_max")
  if (!is.null(x)) {
    x <- x * (i_max - i_min) + i_min
    return(x)
  }
  else {
    data <- data * (i_max - i_min) + i_min
    attr(data, "i_min") <- i_min
    attr(data, "i_max") <- i_max
    return(data)
  }
}
