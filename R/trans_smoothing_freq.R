#'@title Smoothing by equal frequency
#'@description Discretize a numeric vector into `n` bins with approximately equal frequency (quantile cuts),
#' and replace each value by the mean of its bin.
#'@param n number of bins
#'@return returns an object of class `smoothing_freq`
#'@references
#' Han, J., Kamber, M., Pei, J. (2011). Data Mining: Concepts and Techniques. (Discretization)
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
#'@exportS3Method fit smoothing_freq
fit.smoothing_freq <- function(obj, data, ...) {
  if (length(obj$n) > 1)
    obj <- obj$tune(obj, data)
  else {
    v <- data
    n <- obj$n
    # split by quantiles at equal frequency
    p <- seq(from = 0, to = 1, by = 1/n)
    obj$interval <- stats::quantile(v, p)
    obj <- fit.smoothing(obj, data)
  }
  return(obj)
}

