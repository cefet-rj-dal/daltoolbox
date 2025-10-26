#'@title DAL Learner (base class)
#'@description Base ancestor for learning tasks (classification, regression, clustering, time series).
#' Provides common behavior such as proxying `action()` to the model‑specific operation
#' (e.g., `predict()` for predictors, `cluster()` for clusterers) and an `evaluate()` generic.
#'
#' An example of a learner is a decision tree (see `cla_dtree`).
#'@return returns a learner object
#'@examples
#'#See ?cla_dtree for a classification example using a decision tree
#'@export
dal_learner <- function() {
  obj <- dal_base()
  class(obj) <- append("dal_learner", class(obj))
  return(obj)
}

#'@exportS3Method action dal_learner
action.dal_learner <- function(obj, ...) {
  thiscall <- match.call(expand.dots = TRUE)
  # action() on learners proxies to predict()
  thiscall[[1]] <- as.name("predict")
  result <- eval.parent(thiscall)
  return(result)
}

#'@title Evaluate
#'@description Evaluate learner performance.
#' The actual evaluate varies according to the type of learner (clustering, classification, regression, time series regression)
#'@param obj object
#'@param ... optional arguments
#'@return returns the evaluation
#'@examples
#'data(iris)
#'slevels <- levels(iris$Species)
#classification learner using decision tree
#'model <- cla_dtree("Species", slevels)
#'model <- fit(model, iris)
#'prediction <- predict(model, iris)
# categorical mapping for predictand
#'predictand <- adjust_class_label(iris[,"Species"])
#'test_eval <- evaluate(model, predictand, prediction)
#'test_eval$metrics
#'@export
evaluate <- function(obj, ...) {
  UseMethod("evaluate")
}

#'@exportS3Method evaluate default
evaluate.default <- function(obj, ...) {
  return(NULL)
}

