#'@title CART (rpart)
#'@description Classification tree using `rpart::rpart`.
#'@param attribute target attribute name
#'@return returns a `cla_rpart` object
#'@examples
#'if (requireNamespace("rpart", quietly = TRUE)) {
#'  data(iris)
#'  model <- cla_rpart("Species")
#'  model <- fit(model, iris)
#'  pred <- predict(model, iris)
#'  table(pred, iris$Species)
#'}
#'@export
cla_rpart <- function(attribute) {
  obj <- dal_learner()
  obj$attribute <- attribute
  obj$model <- NULL
  class(obj) <- append("cla_rpart", class(obj))
  return(obj)
}

#'@exportS3Method fit cla_rpart
fit.cla_rpart <- function(obj, data, ...) {
  if (!requireNamespace("rpart", quietly = TRUE)) {
    stop("cla_rpart requires the 'rpart' package. Install with install.packages('rpart').")
  }
  data <- adjust_data.frame(data)
  attr <- obj$attribute
  formula <- stats::formula(paste(attr, "~ ."))
  obj$model <- rpart::rpart(formula, data = data, method = "class")
  obj$levels <- levels(data[[attr]])
  return(obj)
}

#'@importFrom stats predict
#'@exportS3Method predict cla_rpart
predict.cla_rpart <- function(object, newdata, ...) {
  newdata <- adjust_data.frame(newdata)
  pred <- stats::predict(object$model, newdata = newdata, type = "class")
  factor(pred, levels = object$levels)
}
