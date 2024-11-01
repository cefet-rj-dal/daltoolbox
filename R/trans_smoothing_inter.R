#'@title Smoothing by interval
#'@description The "smoothing by interval" function is used to apply a smoothing technique to a vector or time series data using a moving window approach.
#'@param n number of bins
#'@return returns an object of class `smoothing_inter`
#'@examples
#'data(iris)
#'obj <- smoothing_inter(n = 2)
#'obj <- fit(obj, iris$Sepal.Length)
#'sl.bi <- transform(obj, iris$Sepal.Length)
#'table(sl.bi)
#'obj$interval
#'
#'entro <- evaluate(obj, as.factor(names(sl.bi)), iris$Species)
#'entro$entropy
#'@export
smoothing_inter <- function(n) {
  obj <- smoothing(n)
  class(obj) <- append("smoothing_inter", class(obj))
  return(obj)
}

#'@importFrom graphics boxplot
#'@export
fit.smoothing_inter <- function(obj, data, ...) {
  if (length(obj$n) > 1)
    obj <- obj$tune(obj, data)
  else {
    v <- data
    n <- obj$n
    bp <- graphics::boxplot(v, range=1.5, plot = FALSE)
    bimax <- bp$stats[5]
    bimin <- bp$stats[1]
    if (bimin == bimax) {
      bimax = max(v)
      bimin = min(v)
    }
    obj$interval <- seq(from = bimin, to = bimax, by = (bimax-bimin)/n)
    obj <- fit.smoothing(obj, data)
  }
  return(obj)
}

