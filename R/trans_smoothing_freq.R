#'@title Smoothing by Freq
#'@description The 'smoothing_freq' function is used to smooth a given time series data by aggregating observations within a fixed frequency.
#'@param n number of bins
#'@return returns an object of class `smoothing_freq`
#'@examples
#'data(iris)
#'obj <- smoothing_freq(n = 2)
#'obj <- fit(obj, iris$Sepal.Length)
#'sl.bi <- transform(obj, iris$Sepal.Length)
#'table(sl.bi)
#'obj$interval
#'
#'entro <- evaluate(obj, as.factor(names(sl.bi)), iris$Species)
#'entro$entropy
#'@export
smoothing_freq <- function(n) {
  obj <- smoothing(n)
  class(obj) <- append("smoothing_freq", class(obj))
  return(obj)
}

#'@importFrom stats quantile
#'@export
fit.smoothing_freq <- function(obj, data, ...) {
  if (length(obj$n) > 1)
    obj <- obj$tune(obj, data)
  else {
    v <- data
    n <- obj$n
    p <- seq(from = 0, to = 1, by = 1/n)
    obj$interval <- stats::quantile(v, p)
    obj <- fit.smoothing(obj, data)
  }
  return(obj)
}

