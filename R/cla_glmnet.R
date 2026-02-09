#'@title LASSO logistic regression (glmnet)
#'@description Logistic regression with L1 penalty using `glmnet::cv.glmnet`.
#'@param attribute target attribute name (binary)
#'@param lambda which lambda to use ("lambda.min" or "lambda.1se")
#'@return returns a `cla_glmnet` object
#'@examples
#'if (requireNamespace("glmnet", quietly = TRUE)) {
#'  data(iris)
#'  iris_bin <- iris
#'  iris_bin$IsVersicolor <- ifelse(iris_bin$Species == "versicolor", 1, 0)
#'  model <- cla_glmnet("IsVersicolor")
#'  model <- fit(model, iris_bin)
#'  pred <- predict(model, iris_bin)
#'  table(pred, iris_bin$IsVersicolor)
#'}
#'@export
cla_glmnet <- function(attribute, lambda = c("lambda.min", "lambda.1se")) {
  obj <- dal_learner()
  obj$attribute <- attribute
  obj$lambda <- match.arg(lambda)
  obj$model <- NULL
  class(obj) <- append("cla_glmnet", class(obj))
  return(obj)
}

#'@exportS3Method fit cla_glmnet
fit.cla_glmnet <- function(obj, data, ...) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("cla_glmnet requires the 'glmnet' package. Install with install.packages('glmnet').")
  }
  data <- adjust_data.frame(data)
  attr <- obj$attribute
  if (is.null(attr) || !attr %in% names(data)) {
    stop("cla_glmnet: attribute not found in data.")
  }
  x <- data.matrix(data[, setdiff(names(data), attr), drop = FALSE])
  y_raw <- data[[attr]]
  if (is.factor(y_raw) || is.character(y_raw)) {
    y_fac <- factor(y_raw)
    obj$levels <- levels(y_fac)
    y <- as.numeric(y_fac) - 1
  } else {
    y <- as.numeric(y_raw)
    obj$levels <- sort(unique(y))
  }
  obj$model <- glmnet::cv.glmnet(x, y, family = "binomial", alpha = 1)
  return(obj)
}

#'@exportS3Method predict cla_glmnet
predict.cla_glmnet <- function(object, newdata, ...) {
  newdata <- adjust_data.frame(newdata)
  x <- data.matrix(newdata[, setdiff(names(newdata), object$attribute), drop = FALSE])
  prob <- as.numeric(stats::predict(object$model, newx = x, s = object$lambda, type = "response"))
  pred <- ifelse(prob >= 0.5, object$levels[2], object$levels[1])
  if (is.character(object$levels) || is.factor(object$levels)) {
    return(factor(pred, levels = object$levels))
  }
  pred
}
