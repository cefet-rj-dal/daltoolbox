#'@title K-Nearest Neighbors (KNN) Regression
#'@description KNN regression using `FNN::knn.reg`, predicting by averaging the targets of the k nearest neighbors.
#'@details Non‑parametric approach suitable for local smoothing. Sensitive to feature scaling; consider normalization beforehand.
#'@param attribute attribute target to model building
#'@param k number of k neighbors
#'@return returns a knn regression object
#'@references
#' Altman, N. (1992). An Introduction to Kernel and Nearest-Neighbor Nonparametric Regression.
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
#'@exportS3Method fit reg_knn
fit.reg_knn <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  # record features used to train KNN regressor
  obj <- fit.predictor(obj, data)

  # convert features to matrix as required by FNN
  x <- as.matrix(data[,obj$x])
  y <- data[,obj$attribute]

  obj$model <- list(x=x, y=y, k=obj$k)

  return(obj)
}

#'@importFrom FNN knn.reg
#'@exportS3Method predict reg_knn
predict.reg_knn  <- function(object, x, ...) {
  #develop from FNN https://daviddalpiaz.github.io/r4sl/knn-reg.html
  x <- adjust_data.frame(x)
  # ensure numeric matrix with same columns used in training
  x <- as.matrix(x[,object$x])
  prediction <- FNN::knn.reg(train = object$model$x, test = x, y = object$model$y, k = object$model$k)
  return(prediction$pred)
}
