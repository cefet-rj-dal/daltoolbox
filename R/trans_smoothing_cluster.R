#'@title Smoothing by cluster
#'@description Uses clustering method to perform data smoothing.
#' The input vector is divided into clusters using the k-means algorithm.
#' The mean of each cluster is then calculated and used as the
#' smoothed value for all observations within that cluster.
#'@param n number of bins
#'@return returns an object of class `smoothing_cluster`
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
