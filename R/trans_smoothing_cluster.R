#'@title Smoothing by clustering (k-means)
#'@description Quantize a numeric vector into `n` levels using k‑means on the values and
#' replace each value by its cluster mean (vector quantization).
#'@param n number of bins
#'@return returns an object of class `smoothing_cluster`
#'@references
#' MacQueen, J. (1967). Some Methods for classification and Analysis of Multivariate Observations.
#'@examples
#'data(iris)
#'obj <- smoothing_cluster(n = 2)
#'obj <- fit(obj, iris$Sepal.Length)
#'sl.bi <- transform(obj, iris$Sepal.Length)
#'table(sl.bi)
#'obj$interval
#'
#'entro <- evaluate(obj, as.factor(names(sl.bi)), iris$Species)
#'entro$entropy
#'@export
smoothing_cluster <- function(n) {
  obj <- smoothing(n)
  class(obj) <- append("smoothing_cluster", class(obj))
  return(obj)
}

#'@importFrom stats kmeans
#'@exportS3Method fit smoothing_cluster
fit.smoothing_cluster <- function(obj, data, ...) {
  if (length(obj$n) > 1)
    obj <- obj$tune(obj, data)
  else {
    v <- data
    n <- obj$n
    # cluster the values and derive cut points from cluster centers
    km <- stats::kmeans(x = v, centers = n)
    s <- sort(km$centers)
    # midpoints between sorted centers define bin boundaries
    s <- (s[1:n-1]+s[2:n])/2
    obj$interval <- c(min(v), s, max(v))
    obj <- fit.smoothing(obj, data)
  }
  return(obj)
}
