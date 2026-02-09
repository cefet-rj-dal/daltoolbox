#'@title Logistic regression (GLM)
#'@description Logistic regression classifier using `stats::glm` with binomial family.
#'@param attribute target attribute name
#'@param positive positive class label
#'@param features optional vector of feature names (default: all except attribute)
#'@param threshold probability threshold for positive class
#'@return returns a `cla_glm` object
#'@examples
#'data(iris)
#'iris_bin <- iris
#'iris_bin$IsVersicolor <- factor(ifelse(
#'  iris_bin$Species == "versicolor",
#'  "versicolor",
#'  "not_versicolor"
#'))
#'model <- cla_glm("IsVersicolor", positive = "versicolor")
#'model <- suppressWarnings(fit(model, iris_bin))
#'pred <- predict(model, iris_bin)
#'table(pred, iris_bin$IsVersicolor)
#'@export
cla_glm <- function(attribute, positive, features = NULL, threshold = 0.5) {
  obj <- dal_learner()
  obj$attribute <- attribute
  obj$positive <- positive
  obj$features <- features
  obj$threshold <- threshold
  obj$model <- NULL
  class(obj) <- append("cla_glm", class(obj))
  return(obj)
}

#'@importFrom stats glm binomial
#'@exportS3Method fit cla_glm
fit.cla_glm <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  attr <- obj$attribute
  if (is.null(attr) || !attr %in% names(data)) {
    stop("cla_glm: attribute not found in data.")
  }
  features <- obj$features
  if (is.null(features)) {
    features <- setdiff(names(data), attr)
  }
  formula <- stats::formula(
    paste(attr, "~", paste(features, collapse = " + "))
  )
  obj$model <- stats::glm(formula, data = data, family = binomial)
  obj$levels <- levels(data[[attr]])
  return(obj)
}

#'@importFrom stats predict
#'@exportS3Method predict cla_glm
predict.cla_glm <- function(object, newdata, ...) {
  newdata <- adjust_data.frame(newdata)
  prob <- stats::predict(object$model, newdata = newdata, type = "response")
  pred <- ifelse(prob >= object$threshold, object$positive, setdiff(object$levels, object$positive)[1])
  factor(pred, levels = object$levels)
}
