#'@title maximum curvature analysis
#'@description Fitting a curvature model in a sequence of observations. It extracts the the maximum curvature computed.
#'@return returns an object of class fit_curvature_max, which inherits from the fit_curvature and dal_transform classes.
#' The object contains a list with the following elements:
#' \itemize{
#' \item x: The position in which the maximum curvature is reached.
#' \item y: The value where the the maximum curvature occurs.
#' \item yfit: The value of the maximum curvature.
#' }
#'
#'@examples
#'x <- seq(from=1,to=10,by=0.5)
#'dat <- data.frame(x = x, value = -log(x), variable = "log")
#'myfit <- fit_curvature_max()
#'res <- transform(myfit, dat$value)
#'head(res)
#'@export
fit_curvature_max <- function() {
  obj <- dal_transform()
  obj$df <- 2
  obj$deriv <- 2
  class(obj) <- append("fit_curvature_max", class(obj))
  return(obj)
}


#'@importFrom stats predict
#'@importFrom stats smooth.spline
#'@export
transform.fit_curvature_max <- function(obj, y, ...) {
  x <- 1:length(y)
  smodel = stats::smooth.spline(x, y, df = obj$df)
  curvature = stats::predict(smodel, x = x, deriv = obj$deriv)
  yfit = max(curvature$y)
  xfit = match(yfit, curvature$y)
  y <- y[xfit]
  res <- data.frame(x=xfit, y=y, yfit = yfit)
  return(res)
}


