#'@title Bagging (ipred)
#'@description Bagging classifier using `ipred::bagging`.
#'@param attribute target attribute name
#'@param nbagg number of bootstrap aggregations
#'@return returns a `cla_bagging` object
#'@examples
#'if (requireNamespace("ipred", quietly = TRUE)) {
#'  data(iris)
#'  model <- cla_bagging("Species", nbagg = 25)
#'  model <- fit(model, iris)
#'  pred <- predict(model, iris)
#'  table(pred, iris$Species)
#'}
#'@export
cla_bagging <- function(attribute, nbagg = 25) {
  obj <- dal_learner()
  obj$attribute <- attribute
  obj$nbagg <- nbagg
  obj$model <- NULL
  class(obj) <- append("cla_bagging", class(obj))
  return(obj)
}

#'@exportS3Method fit cla_bagging
fit.cla_bagging <- function(obj, data, ...) {
  if (!requireNamespace("ipred", quietly = TRUE)) {
    stop("cla_bagging requires the 'ipred' package. Install with install.packages('ipred').")
  }
  data <- adjust_data.frame(data)
  attr <- obj$attribute
  formula <- stats::formula(paste(attr, "~ ."))
  obj$model <- ipred::bagging(formula, data = data, nbagg = obj$nbagg)
  obj$levels <- levels(data[[attr]])
  return(obj)
}

#'@importFrom stats predict
#'@exportS3Method predict cla_bagging
predict.cla_bagging <- function(object, newdata, ...) {
  newdata <- adjust_data.frame(newdata)
  pred <- stats::predict(object$model, newdata = newdata, type = "class")
  factor(pred, levels = object$levels)
}
