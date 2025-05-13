#'@title no normalization
#'@description Does not make data normalization.
#'@return a `ts_norm_none` object.
#'@examples
#'library(daltoolbox)
#'data(sin_data)
#'
#'#convert to sliding windows
#'xw <- ts_data(sin_data$y, 10)
#'
#'#no data normalization
#'normalize <- ts_norm_none()
#'normalize <- fit(normalize, xw)
#'xa <- transform(normalize, xw)
#'ts_head(xa)
#'@importFrom daltoolbox dal_transform
#'@importFrom daltoolbox fit
#'@importFrom daltoolbox transform
#'@export
ts_norm_none <- function() {
  obj <- dal_transform()
  class(obj) <- append("ts_norm_none", class(obj))
  return(obj)
}

#'@importFrom daltoolbox transform
#'@exportS3Method transform ts_norm_none
transform.ts_norm_none <- function(obj, data, ...) {
  result <- data
  idx <- c(1:nrow(result))
  attr(result, "idx") <- idx
  return(result)
}

