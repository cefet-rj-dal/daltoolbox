#'@title Boosting (adabag)
#'@description Boosting classifier using `adabag::boosting`.
#'@param attribute target attribute name
#'@param mfinal number of boosting iterations
#'@return returns a `cla_boosting` object
#'@examples
#'if (requireNamespace("adabag", quietly = TRUE)) {
#'  data(iris)
#'  model <- cla_boosting("Species", mfinal = 10)
#'  model <- fit(model, iris)
#'  pred <- predict(model, iris)
#'  table(pred, iris$Species)
#'}
#'@export
cla_boosting <- function(attribute, mfinal = 50) {
  obj <- dal_learner()
  obj$attribute <- attribute
  obj$mfinal <- mfinal
  obj$model <- NULL
  class(obj) <- append("cla_boosting", class(obj))
  return(obj)
}

#'@exportS3Method fit cla_boosting
fit.cla_boosting <- function(obj, data, ...) {
  if (!requireNamespace("adabag", quietly = TRUE)) {
    stop("cla_boosting requires the 'adabag' package. Install with install.packages('adabag').")
  }
  data <- adjust_data.frame(data)
  attr <- obj$attribute
  formula <- stats::formula(paste(attr, "~ ."))
  obj$model <- adabag::boosting(formula, data = data, mfinal = obj$mfinal)
  obj$levels <- levels(data[[attr]])
  return(obj)
}

#'@importFrom stats predict
#'@exportS3Method predict cla_boosting
predict.cla_boosting <- function(object, newdata, ...) {
  newdata <- adjust_data.frame(newdata)
  pred <- stats::predict(object$model, newdata = newdata)$class
  factor(pred, levels = object$levels)
}
