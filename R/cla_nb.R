#'@title Naive Bayes Classifier
#'@description Naive Bayes classification using `e1071::naiveBayes`.
#'@details Assumes conditional independence of features given the class label, enabling fast probabilistic classification.
#'@param attribute attribute target to model building.
#'@param slevels possible values for the target classification.
#'@return returns a classification object.
#'@references
#' Mitchell, T. (1997). Machine Learning. McGraw‑Hill. (Naive Bayes)
#'@examples
#'data(iris)
#'slevels <- levels(iris$Species)
#'model <- cla_nb("Species", slevels)
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
cla_nb <- function(attribute, slevels) {
  obj <- classification(attribute, slevels)

  class(obj) <- append("cla_nb", class(obj))
  return(obj)
}

#'@importFrom e1071 naiveBayes
#'@exportS3Method fit cla_nb
fit.cla_nb <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  # ensure factor target with expected label set
  data[,obj$attribute] <- adjust_factor(data[,obj$attribute], obj$ilevels, obj$slevels)
  obj <- fit.predictor(obj, data)

  # build formula target ~ .
  regression <- formula(paste(obj$attribute, "  ~ ."))
  obj$model <- e1071::naiveBayes(regression, data, laplace=0)


  return(obj)
}

#'@exportS3Method predict cla_nb
predict.cla_nb  <- function(object, x, ...) {
  x <- adjust_data.frame(x)
  x <- x[,object$x, drop=FALSE]

  prediction <- predict(object$model, x, type="raw")
  prediction <- as.data.frame(prediction)
  colnames(prediction) <- object$slevels

  return(prediction)
}
