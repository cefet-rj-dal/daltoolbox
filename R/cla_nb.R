#'@title Naive Bayes Classifier
#'@description Classification using the Naive Bayes algorithm
#' It wraps the e1071 library.
#'@param attribute attribute target to model building.
#'@param slevels possible values for the target classification.
#'@return returns a classification object.
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

#'@import e1071
#'@export
fit.cla_nb <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  data[,obj$attribute] <- adjust_factor(data[,obj$attribute], obj$ilevels, obj$slevels)
  obj <- fit.predictor(obj, data)

  regression <- formula(paste(obj$attribute, "  ~ ."))
  obj$model <- e1071::naiveBayes(regression, data, laplace=0)


  return(obj)
}

#'@export
predict.cla_nb  <- function(object, x, ...) {
  x <- adjust_data.frame(x)
  x <- x[,object$x, drop=FALSE]

  prediction <- predict(object$model, x, type="raw")

  return(prediction)
}
