#'@title knn regression
#'@description Creates a regression object that
#' uses the K-Nearest Neighbors (knn) method for regression
#'@param attribute attribute target to model building
#'@param k number of k neighbors
#'@return returns a knn regression object
#'@examples
#'data(Boston)
#'model <- reg_knn("medv", k=3)
#'
#'# preparing dataset for random sampling
#'sr <- sample_random()
#'sr <- train_test(sr, Boston)
#'train <- sr$train
#'test <- sr$test
#'
#'model <- fit(model, train)
#'
#'test_prediction <- predict(model, test)
#'test_predictand <- test[,"medv"]
#'test_eval <- evaluate(model, test_predictand, test_prediction)
#'test_eval$metrics
#'@export
reg_knn <- function(attribute, k) {
  obj <- regression(attribute)
  obj$k <- k

  class(obj) <- append("reg_knn", class(obj))
  return(obj)
}

#'@importFrom FNN knn.reg
#'@export
fit.reg_knn <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  obj <- fit.predictor(obj, data)

  x <- as.matrix(data[,obj$x])
  y <- data[,obj$attribute]

  obj$model <- list(x=x, y=y, k=obj$k)

  return(obj)
}

#'@importFrom FNN knn.reg
#'@export
predict.reg_knn  <- function(object, x, ...) {
  #develop from FNN https://daviddalpiaz.github.io/r4sl/knn-reg.html
  x <- adjust_data.frame(x)
  x <- as.matrix(x[,object$x])
  prediction <- FNN::knn.reg(train = object$model$x, test = x, y = object$model$y, k = object$model$k)
  return(prediction$pred)
}
