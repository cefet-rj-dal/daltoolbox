#'@title Majority Classification
#'@description This function creates a classification object that uses the majority vote strategy to predict the target attribute. Given a target attribute, the function counts the number of occurrences of each value in the dataset and selects the one that appears most often.
#'@param attribute attribute target to model building.
#'@param slevels possible values for the target classification.
#'@return returns a classification object.
#'@examples
#'data(iris)
#'slevels <- levels(iris$Species)
#'model <- cla_majority("Species", slevels)
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
cla_majority <- function(attribute, slevels) {
  obj <- classification(attribute, slevels)

  class(obj) <- append("cla_majority", class(obj))
  return(obj)
}

#'@exportS3Method fit cla_majority
fit.cla_majority <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  data[,obj$attribute] <- adjust_factor(data[,obj$attribute], obj$ilevels, obj$slevels)
  obj <- fit.predictor(obj, data)

  y <- adjust_class_label(data[,obj$attribute])
  cols <- apply(y, 2, sum)
  col <- match(max(cols),cols)
  obj$model <- list(cols=cols, col=col)

  return(obj)
}

#'@exportS3Method predict cla_majority
predict.cla_majority <- function(object, x, ...) {
  rows <- nrow(x)
  cols <- length(object$model$cols)
  prediction <- matrix(rep.int(0, rows*cols), nrow=rows, ncol=cols)
  prediction[,object$model$col] <- 1
  prediction <- as.data.frame(prediction)
  colnames(prediction) <- object$slevels
  return(prediction)
}


