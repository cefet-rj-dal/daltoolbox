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
  obj <- classification(attribute)
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
  prepared <- prepare_classification_data(obj, data)
  obj <- prepared$obj
  data <- prepared$data
  if (length(obj$slevels) != 2) {
    stop("cla_glmnet supports only binary classification.", call. = FALSE)
  }
  x <- data.matrix(data[, obj$x, drop = FALSE])
  y <- as.numeric(data[[obj$attribute]]) - 1
  obj$model <- glmnet::cv.glmnet(x, y, family = "binomial", alpha = 1)
  return(obj)
}

#'@exportS3Method predict cla_glmnet
predict.cla_glmnet <- function(object, newdata, ...) {
  newdata <- adjust_data.frame(newdata)
  x <- data.matrix(newdata[, object$x, drop = FALSE])
  prob <- as.numeric(stats::predict(object$model, newx = x, s = object$lambda, type = "response"))
  prediction <- data.frame(
    setNames(list(1 - prob), object$slevels[1]),
    setNames(list(prob), object$slevels[2]),
    check.names = FALSE
  )
  prediction
}
