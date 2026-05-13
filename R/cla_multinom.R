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
#'eval <- evaluate(model, adjust_class_label(iris$Species), pred)
#'eval$metrics
#'@export
cla_multinom <- function(attribute, features = NULL) {
  obj <- classification(attribute)
  obj$features <- features
  obj$model <- NULL
  class(obj) <- append("cla_multinom", class(obj))
  return(obj)
}

#'@importFrom nnet multinom
#'@exportS3Method fit cla_multinom
fit.cla_multinom <- function(obj, data, ...) {
  prepared <- prepare_classification_data(obj, data)
  obj <- prepared$obj
  data <- prepared$data
  attr <- obj$attribute
  features <- obj$features
  if (is.null(features)) {
    features <- obj$x
  }
  formula <- stats::formula(
    paste(attr, "~", paste(features, collapse = " + "))
  )
  obj$model <- nnet::multinom(formula, data = data, trace = FALSE)
  return(obj)
}

#'@importFrom stats predict
#'@exportS3Method predict cla_multinom
predict.cla_multinom <- function(object, newdata, ...) {
  newdata <- adjust_data.frame(newdata)
  x <- newdata[, object$x, drop = FALSE]
  prediction <- stats::predict(object$model, newdata = x, type = "probs")
  if (is.null(dim(prediction))) {
    prediction <- cbind(1 - prediction, prediction)
  }
  prediction <- as.matrix(prediction)
  colnames(prediction) <- object$slevels
  as.data.frame(prediction)
}
