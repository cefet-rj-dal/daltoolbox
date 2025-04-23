#'@title DAL Predict
#'@description Ancestor class for regression and classification
#'It provides basis for fit and predict methods.
#'Besides, action method proxies to predict.
#'
#' An example of learner is a decision tree (cla_dtree)
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
  obj$x <- setdiff(colnames(data), obj$attribute)
  return(obj)
}

#'@exportS3Method action predictor
action.predictor <- function(obj, ...) {
  thiscall <- match.call(expand.dots = TRUE)
  thiscall[[1]] <- as.name("predict")
  result <- eval.parent(thiscall)
  return(result)
}

