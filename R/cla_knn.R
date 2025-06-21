#'@title K Nearest Neighbor Classification
#'@description Classifies using the K-Nearest Neighbor algorithm.
#' It wraps the class library.
#'@param attribute attribute target to model building.
#'@param slevels possible values for the target classification.
#'@param k a vector of integers indicating the number of neighbors to be considered.
#'@return returns a knn object.
#'@examples
#'data(iris)
#'slevels <- levels(iris$Species)
#'model <- cla_knn("Species", slevels, k=3)
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
cla_knn <- function(attribute, slevels, k=1) {
  obj <- classification(attribute, slevels)
  obj$k <- k
  class(obj) <- append("cla_knn", class(obj))
  return(obj)
}

#'@exportS3Method fit cla_knn
fit.cla_knn <- function(obj, data, ...) {

  data <- adjust_data.frame(data)
  data[,obj$attribute] <- adjust_factor(data[,obj$attribute], obj$ilevels, obj$slevels)
  obj <- fit.predictor(obj, data)

  x <- data[,obj$x, drop = FALSE]
  y <- data[,obj$attribute]

  obj$model <-list(x=x, y=y, k=obj$k)

  return(obj)
}

#'@importFrom class knn
#'@exportS3Method predict cla_knn
predict.cla_knn  <- function(object, x, ...) {
  x <- adjust_data.frame(x)
  x <- x[,object$x, drop=FALSE]

  prediction <- class::knn(train=object$model$x, test=x, cl=object$model$y, prob=TRUE)
  prediction <- adjust_class_label(prediction)
  prediction <- as.data.frame(prediction)
  colnames(prediction) <- object$slevels

  return(prediction)
}


