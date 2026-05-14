#'@title Smoothing by quantization (k-means)
#'@description Quantize a numeric vector into `n` levels using k‑means on the values and
#' replace each value by its cluster mean (vector quantization).
#'@param n number of bins
#'@return returns an object of class `smoothing_quantization`
#'@references
#' MacQueen, J. (1967). Some Methods for classification and Analysis of Multivariate Observations.
#'@examples
#'data(iris)
#'obj <- smoothing_quantization(n = 2)
#'obj <- fit(obj, iris$Sepal.Length)
#'sl.bi <- transform(obj, iris$Sepal.Length)
#'table(sl.bi)
#'obj$interval
#'
#'bins <- cut(iris$Sepal.Length, unique(obj$interval.adj), FALSE, include.lowest = TRUE)
#'entro <- evaluate(obj, bins, iris$Species)
#'entro$entropy
#'@export
smoothing_quantization <- function(n) {
  obj <- smoothing(n)
  class(obj) <- append("smoothing_quantization", class(obj))
  return(obj)
}

#'@importFrom stats kmeans
#'@exportS3Method fit smoothing_quantization
fit.smoothing_quantization <- function(obj, data, ...) {
  if (length(obj$n) > 1)
    obj <- obj$tune(obj, data)
  else {
    v <- data
    n <- obj$n
    # cluster the values and derive cut points from cluster centers
    km <- stats::kmeans(x = v, centers = n)
    s <- sort(km$centers)
    # midpoints between sorted centers define bin boundaries
    s <- if (n > 1) (s[1:n-1] + s[2:n]) / 2 else numeric(0)
    obj$interval <- c(min(v), s, max(v))
    obj <- fit.smoothing(obj, data)
  }
  return(obj)
}
