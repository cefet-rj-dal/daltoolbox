#'@title Minimum curvature analysis (elbow detection)
#'@description Computes a smoothing spline over a sequence and returns the location/value of minimum curvature,
#' complementary to maximum curvature and useful in elbow detection.
#'@return Returns an object of class fit_curvature_max, which inherits from the fit_curvature and dal_transform classes.
#' The object contains a list with the following elements:
#' \itemize{
#' \item x: The position in which the minimum curvature is reached.
#' \item y: The value where the the minimum curvature occurs.
#' \item yfit: The value of the minimum curvature.
#' }
#'@examples
#'x <- seq(from=1,to=10,by=0.5)
#'dat <- data.frame(x = x, value = log(x), variable = "log")
#'myfit <- fit_curvature_min()
#'res <- transform(myfit, dat$value)
#'head(res)
#'@export
fit_curvature_min <- function() {
  obj <- dal_transform()
  obj$df <- 2
  obj$deriv <- 2
  class(obj) <- append("fit_curvature_min", class(obj))
  return(obj)
}

#'@importFrom stats predict
#'@importFrom stats smooth.spline
#'@exportS3Method transform fit_curvature_min
transform.fit_curvature_min <- function(obj, y, ...) {
  x <- 1:length(y)
  # fit smoothing spline and compute curvature (2nd derivative)
  smodel = stats::smooth.spline(x, y, df = obj$df)
  curvature = stats::predict(smodel, x = x, deriv = obj$deriv)
  # pick minimum curvature point
  yfit = min(curvature$y)
  xfit = match(yfit, curvature$y)
  y <- y[xfit]
  res <- data.frame(x=xfit, y=y, yfit = yfit)
  return(res)
}
#'@references
#' Satopaa, V., Albrecht, J., Irwin, D., Raghavan, B. (2011). Finding a “Kneedle” in a Haystack: Detecting Knee Points in System Behavior.
