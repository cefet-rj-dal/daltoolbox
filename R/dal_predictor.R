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

#'@exportS3Method fit predictor
fit.predictor <- function(obj, data, ...) {
  # capture feature names by removing target attribute from columns
  obj$x <- setdiff(colnames(data), obj$attribute)
  return(obj)
}

#'@exportS3Method action predictor
action.predictor <- function(obj, ...) {
  thiscall <- match.call(expand.dots = TRUE)
  # proxy action() to S3 predict()
  thiscall[[1]] <- as.name("predict")
  result <- eval.parent(thiscall)
  return(result)
}

