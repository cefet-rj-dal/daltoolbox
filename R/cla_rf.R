#'@title Random Forest for classification
#'@description Ensemble classifier of decision trees using `randomForest::randomForest`.
#'@details Combines many decorrelated trees to reduce variance. Key hyperparameters: `ntree`, `mtry`, `nodesize`.
#'@param attribute attribute target to model building
#'@param slevels possible values for the target classification
#'@param nodesize node size
#'@param ntree number of trees
#'@param mtry number of attributes to build tree
#'@return returns a classification object
#'@references
#' Breiman, L. (2001). Random Forests. Machine Learning 45(1):5–32.
#' Liaw, A. and Wiener, M. (2002). Classification and Regression by randomForest. R News.
#'@examples
#'data(iris)
#'slevels <- levels(iris$Species)
#'model <- cla_rf("Species", slevels, ntree=5)
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
cla_rf <- function(attribute, slevels, nodesize = 5, ntree = 10, mtry = NULL) {
  obj <- classification(attribute, slevels)

  obj$nodesize <- nodesize
  obj$ntree <- ntree
  obj$mtry <- mtry

  class(obj) <- append("cla_rf", class(obj))
  return(obj)
}

#'@importFrom randomForest randomForest
#'@exportS3Method fit cla_rf
fit.cla_rf <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  # convert target to factor and align levels
  data[,obj$attribute] <- adjust_factor(data[,obj$attribute], obj$ilevels, obj$slevels)
  # record feature set
  obj <- fit.predictor(obj, data)

  # default mtry heuristic for classification
  if (is.null(obj$mtry))
    obj$mtry <- ceiling(sqrt(ncol(data)))

  x <- data[,obj$x, drop=FALSE]
  y <- data[,obj$attribute]

  obj$model <- randomForest::randomForest(x = x, y = y, nodesize = obj$nodesize, mtry=obj$mtry, ntree=obj$ntree)

  return(obj)
}

#'@exportS3Method predict cla_rf
predict.cla_rf  <- function(object, x, ...) {
  x <- adjust_data.frame(x)
  # ensure consistent predictors
  x <- x[,object$x, drop = FALSE]

  prediction <- predict(object$model, x, type="prob")
  prediction <- as.data.frame(prediction)
  colnames(prediction) <- object$slevels

  return(prediction)
}
