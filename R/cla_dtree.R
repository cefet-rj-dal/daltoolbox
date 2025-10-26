#'@title Decision Tree for classification
#'@description Univariate decision tree for classification using recursive partitioning.
#' This wrapper uses the `tree` package.
#'@details Decision trees split the feature space by maximizing node purity (e.g., Gini/entropy),
#' yielding a human‑readable set of rules. They are fast and interpretable, and often used as
#' base learners in ensembles.
#'@param attribute attribute target to model building
#'@param slevels the possible values for the target classification
#'@return returns a classification object
#'@references
#' Breiman, L., Friedman, J., Olshen, R., and Stone, C. (1984).
#' Classification and Regression Trees. Wadsworth.
#'@examples
#'data(iris)
#'slevels <- levels(iris$Species)
#'model <- cla_dtree("Species", slevels)
#'
#'# preparing dataset for random sampling
#'sr <- sample_random()
#'sr <- train_test(sr, iris)
#'train <- sr$train
#'test <- sr$test
#'
#'model <- fit(model, train)
#'
#'prediction <- predict(model, test)
#'predictand <- adjust_class_label(test[,"Species"])
#'test_eval <- evaluate(model, predictand, prediction)
#'test_eval$metrics
#'@export
cla_dtree <- function(attribute, slevels) {
  obj <- classification(attribute, slevels)

  class(obj) <- append("cla_dtree", class(obj))
  return(obj)
}

#'@importFrom tree tree
#'@exportS3Method fit cla_dtree
fit.cla_dtree <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  # coerce target into factor with expected levels/labels
  data[,obj$attribute] <- adjust_factor(data[,obj$attribute], obj$ilevels, obj$slevels)
  # record feature columns (exclude target)
  obj <- fit.predictor(obj, data)

  # build formula target ~ . (all remaining columns)
  regression <- formula(paste(obj$attribute, "  ~ ."))
  obj$model <- tree::tree(regression, data)


  return(obj)
}

#'@exportS3Method predict cla_dtree
predict.cla_dtree <- function(object, x, ...) {
  x <- adjust_data.frame(x)
  # use same feature set used during training
  x <- x[,object$x, drop=FALSE]

  prediction <- predict(object$model, x, type="vector")
  prediction <- as.data.frame(prediction)
  colnames(prediction) <- object$slevels

  return(prediction)
}





