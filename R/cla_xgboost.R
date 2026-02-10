#'@title XGBoost
#'@description Gradient boosting classifier using `xgboost`.
#'@param attribute target attribute name
#'@param params list of xgboost parameters
#'@param nrounds number of boosting rounds
#'@return returns a `cla_xgboost` object
#'@examples
#'if (requireNamespace("xgboost", quietly = TRUE)) {
#'  data(iris)
#'  model <- cla_xgboost("Species", nrounds = 5)
#'  model <- fit(model, iris)
#'  pred <- predict(model, iris)
#'  table(pred, iris$Species)
#'}
#'@export
cla_xgboost <- function(attribute, params = list(), nrounds = 20) {
  obj <- dal_learner()
  obj$attribute <- attribute
  obj$params <- params
  obj$nrounds <- nrounds
  obj$model <- NULL
  obj$levels <- NULL
  class(obj) <- append("cla_xgboost", class(obj))
  return(obj)
}

#'@exportS3Method fit cla_xgboost
fit.cla_xgboost <- function(obj, data, ...) {
  if (!requireNamespace("xgboost", quietly = TRUE)) {
    stop("cla_xgboost requires the 'xgboost' package. Install with install.packages('xgboost').")
  }
  data <- adjust_data.frame(data)
  attr <- obj$attribute
  if (is.null(attr) || !attr %in% names(data)) {
    stop("cla_xgboost: attribute not found in data.")
  }
  x <- as.matrix(data[, setdiff(names(data), attr), drop = FALSE])
  y <- data[[attr]]
  obj$levels <- levels(y)
  y_num <- as.numeric(y) - 1
  params <- obj$params
  if (length(obj$levels) == 2) {
    params$objective <- if (is.null(params$objective)) "binary:logistic" else params$objective
    dtrain <- xgboost::xgb.DMatrix(data = x, label = y_num)
  } else {
    params$objective <- if (is.null(params$objective)) "multi:softprob" else params$objective
    params$num_class <- if (is.null(params$num_class)) length(obj$levels) else params$num_class
    dtrain <- xgboost::xgb.DMatrix(data = x, label = y_num)
  }
  obj$model <- xgboost::xgb.train(params = params, data = dtrain, nrounds = obj$nrounds, verbose = 0)
  return(obj)
}

#'@exportS3Method predict cla_xgboost
predict.cla_xgboost <- function(object, newdata, ...) {
  newdata <- adjust_data.frame(newdata)
  x <- as.matrix(newdata[, setdiff(names(newdata), object$attribute), drop = FALSE])
  dtest <- xgboost::xgb.DMatrix(data = x)
  preds <- predict(object$model, dtest)
  if (length(object$levels) == 2) {
    pred <- ifelse(preds >= 0.5, object$levels[2], object$levels[1])
  } else {
    probs <- matrix(preds, ncol = length(object$levels), byrow = TRUE)
    pred <- object$levels[max.col(probs, ties.method = "first")]
  }
  factor(pred, levels = object$levels)
}
