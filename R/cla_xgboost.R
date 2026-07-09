#'@title XGBoost
#'@description Gradient boosting classifier using `xgboost`.
#'@param attribute target attribute name
#'@param params list of xgboost parameters
#'@param nrounds number of boosting rounds
#'@return returns a `cla_xgboost` object
#'@examples
#'\dontrun{
#'data(iris)
#'model <- cla_xgboost("Species")
#'model <- fit(model, iris)
#'pred <- predict(model, iris)
#'eval <- evaluate(model, adjust_class_label(iris$Species), pred)
#'eval$metrics
#'}
#'@export
cla_xgboost <- function(attribute, params = list(), nrounds = 20) {
  obj <- classification(attribute)
  obj$params <- params
  obj$nrounds <- nrounds
  obj$model <- NULL
  class(obj) <- append("cla_xgboost", class(obj))
  return(obj)
}

#'@exportS3Method fit cla_xgboost
fit.cla_xgboost <- function(obj, data, ...) {
  if (!requireNamespace("xgboost", quietly = TRUE)) {
    stop("cla_xgboost requires the 'xgboost' package. Install with install.packages('xgboost').")
  }
  prepared <- prepare_classification_data(obj, data)
  obj <- prepared$obj
  data <- prepared$data
  x <- as.matrix(data[, obj$x, drop = FALSE])
  y_num <- as.numeric(data[[obj$attribute]]) - 1
  params <- obj$params
  if (length(obj$slevels) == 2) {
    params$objective <- if (is.null(params$objective)) "binary:logistic" else params$objective
    dtrain <- xgboost::xgb.DMatrix(data = x, label = y_num)
  } else {
    params$objective <- if (is.null(params$objective)) "multi:softprob" else params$objective
    params$num_class <- if (is.null(params$num_class)) length(obj$slevels) else params$num_class
    dtrain <- xgboost::xgb.DMatrix(data = x, label = y_num)
  }
  obj$model <- xgboost::xgb.train(params = params, data = dtrain, nrounds = obj$nrounds, verbose = 0)
  return(obj)
}

#'@exportS3Method predict cla_xgboost
predict.cla_xgboost <- function(object, newdata, ...) {
  newdata <- adjust_data.frame(newdata)
  x <- as.matrix(newdata[, object$x, drop = FALSE])
  dtest <- xgboost::xgb.DMatrix(data = x)
  preds <- predict(object$model, dtest)
  if (length(object$slevels) == 2) {
    if (is.matrix(preds)) {
      prediction <- preds
    } else {
      prediction <- cbind(1 - preds, preds)
    }
  } else {
    if (is.matrix(preds)) {
      prediction <- preds
    } else {
      prediction <- matrix(preds, ncol = length(object$slevels), byrow = TRUE)
    }
  }
  prediction <- as.data.frame(prediction)
  colnames(prediction) <- object$slevels
  prediction
}
