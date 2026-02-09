#'@title Multinomial logistic regression
#'@description Multiclass classification using `nnet::multinom`.
#'@param attribute target attribute name
#'@param features optional vector of feature names (default: all except attribute)
#'@return returns a `cla_multinom` object
#'@examples
#'data(iris)
#'model <- cla_multinom("Species")
#'model <- fit(model, iris)
#'pred <- predict(model, iris)
#'table(pred, iris$Species)
#'@export
cla_multinom <- function(attribute, features = NULL) {
  obj <- dal_learner()
  obj$attribute <- attribute
  obj$features <- features
  obj$model <- NULL
  class(obj) <- append("cla_multinom", class(obj))
  return(obj)
}

#'@importFrom nnet multinom
#'@exportS3Method fit cla_multinom
fit.cla_multinom <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  attr <- obj$attribute
  if (is.null(attr) || !attr %in% names(data)) {
    stop("cla_multinom: attribute not found in data.")
  }
  features <- obj$features
  if (is.null(features)) {
    features <- setdiff(names(data), attr)
  }
  formula <- stats::formula(
    paste(attr, "~", paste(features, collapse = " + "))
  )
  obj$model <- nnet::multinom(formula, data = data, trace = FALSE)
  obj$levels <- levels(data[[attr]])
  return(obj)
}

#'@importFrom stats predict
#'@exportS3Method predict cla_multinom
predict.cla_multinom <- function(object, newdata, ...) {
  newdata <- adjust_data.frame(newdata)
  pred <- stats::predict(object$model, newdata = newdata)
  factor(pred, levels = object$levels)
}
