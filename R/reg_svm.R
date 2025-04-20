#'@title SVM for regression
#'@description Creates a regression object that
#' uses the Support Vector Machine (SVM) method for regression
#' It wraps the e1071 and svm library.
#'@param attribute attribute target to model building
#'@param epsilon parameter that controls the width of the margin around the separating hyperplane
#'@param cost parameter that controls the trade-off between having a wide margin and correctly classifying training data points
#'@param kernel the type of kernel function to be used in the SVM algorithm (linear, radial, polynomial, sigmoid)
#'@return returns a SVM regression object
#'@examples
#'data(Boston)
#'model <- reg_svm("medv", epsilon=0.2,cost=40.000)
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
reg_svm <- function(attribute, epsilon=0.1, cost=10, kernel="radial") {
  #analisar: https://rpubs.com/Kushan/296706
  obj <- regression(attribute)
  obj$kernel <- kernel
  obj$epsilon <- epsilon
  obj$cost <- cost

  class(obj) <- append("reg_svm", class(obj))
  return(obj)
}

#'@importFrom e1071 svm
#'@export
fit.reg_svm <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  obj <- fit.predictor(obj, data)

  x <- data[,obj$x]
  y <- data[,obj$attribute]

  obj$model <- e1071::svm(x = x, y = y, epsilon=obj$epsilon, cost=obj$cost, kernel=obj$kernel)

  return(obj)
}

#'@export
predict.reg_svm  <- function(object, x, ...) {
  x <- adjust_data.frame(x)
  x <- x[,object$x]
  prediction <- predict(object$model, x)
  return(prediction)
}
