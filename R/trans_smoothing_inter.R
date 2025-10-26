#'@title Smoothing by equal interval
#'@description Discretize a numeric vector into `n` equal‑width intervals (robust bounds via boxplot whiskers)
#' and replace each value by the bin mean.
#'@param n number of bins
#'@return returns an object of class `smoothing_inter`
#'@references
#' Han, J., Kamber, M., Pei, J. (2011). Data Mining: Concepts and Techniques. (Discretization)
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
#'@exportS3Method fit smoothing_inter
fit.smoothing_inter <- function(obj, data, ...) {
  if (length(obj$n) > 1)
    obj <- obj$tune(obj, data)
  else {
    v <- data
    n <- obj$n
    # use boxplot whiskers as robust min/max boundaries
    bp <- graphics::boxplot(v, range=1.5, plot = FALSE)
    bimax <- bp$stats[5]
    bimin <- bp$stats[1]
    if (bimin == bimax) {
      bimax = max(v)
      bimin = min(v)
    }
    # create equally spaced intervals across [bimin, bimax]
    obj$interval <- seq(from = bimin, to = bimax, by = (bimax-bimin)/n)
    obj <- fit.smoothing(obj, data)
  }
  return(obj)
}

