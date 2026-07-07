#'@title Predictor (base for classification/regression)
#'@description Ancestor class for supervised predictors (classification and regression).
#' Provides a default `fit()` to record feature names and proxies `action()` to `predict()`.
#'
#' An example predictor is a decision tree classifier (`cla_dtree`).
#'@return returns a predictor object
#'@examples
#'#See ?cla_dtree for a classification example using a decision tree
#'@export
predictor <- function() {
  obj <- dal_learner()
  class(obj) <- append("predictor", class(obj))
  return(obj)
}

#'@title Prepare predictor fit data
#'@description Normalizes supervised fit input and records predictor columns by
#' removing the target attribute from the data columns.
#'@param obj predictor-like object containing `attribute`
#'@param data training data
#'@return returns a list with the updated object and normalized data
#'@export
predictor_prepare_fit <- function(obj, data) {
  if (is.data.frame(data) || is.matrix(data)) {
    data <- adjust_data.frame(data)
  }
  obj$x <- setdiff(colnames(data), obj$attribute)
  list(obj = obj, data = data)
}

#' @method fit predictor
#' @export
fit.predictor <- function(obj, data, ...) {
  predictor_prepare_fit(obj, data)$obj
}

#'@exportS3Method action predictor
action.predictor <- function(obj, ...) {
  thiscall <- match.call(expand.dots = TRUE)
  # proxy action() to S3 predict()
  thiscall[[1]] <- as.name("predict")
  result <- eval.parent(thiscall)
  return(result)
}

