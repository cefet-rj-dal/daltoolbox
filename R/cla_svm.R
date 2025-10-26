#'@title SVM for classification
#'@description Support Vector Machines (SVM) for classification using `e1071::svm`.
#'@details SVMs find a maximum‑margin hyperplane in a transformed feature space defined
#' by a kernel (linear, radial, polynomial, sigmoid). The `cost` controls the trade‑off
#' between margin width and training error; `epsilon` affects stopping; `kernel` sets the feature map.
#'@param attribute attribute target to model building
#'@param slevels possible values for the target classification
#'@param epsilon parameter that controls the width of the margin around the separating hyperplane
#'@param cost parameter that controls the trade-off between having a wide margin and correctly classifying training data points
#'@param kernel the type of kernel function to be used in the SVM algorithm (linear, radial, polynomial, sigmoid)
#'@return returns a SVM classification object
#'@references
#' Cortes, C. and Vapnik, V. (1995). Support-Vector Networks. Machine Learning 20(3):273–297.
#' Chang, C.-C. and Lin, C.-J. (2011). LIBSVM: A library for support vector machines.
#'@examples
#'data(iris)
#'slevels <- levels(iris$Species)
#'model <- cla_svm("Species", slevels, epsilon=0.0,cost=20.000)
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
cla_svm <- function(attribute, slevels, epsilon=0.1, cost=10, kernel="radial") {
  #kernel: linear, radial, polynomial, sigmoid
  #studio: https://rpubs.com/Kushan/296706
  obj <- classification(attribute, slevels)
  obj$kernel <- kernel
  obj$epsilon <- epsilon
  obj$cost <- cost

  class(obj) <- append("cla_svm", class(obj))
  return(obj)
}

#'@importFrom e1071 svm
#'@exportS3Method fit cla_svm
fit.cla_svm <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  # ensure target is a factor with expected levels
  data[,obj$attribute] <- adjust_factor(data[,obj$attribute], obj$ilevels, obj$slevels)
  # capture feature columns used for training
  obj <- fit.predictor(obj, data)

  x <- data[,obj$x, drop=FALSE]
  y <- data[,obj$attribute]

  # enable probability estimates; pass epsilon/cost/kernel
  obj$model <- e1071::svm(x, y, probability=TRUE, epsilon=obj$epsilon, cost=obj$cost, kernel=obj$kernel)

  return(obj)
}

#'@exportS3Method predict cla_svm
predict.cla_svm  <- function(object, x, ...) {
  x <- adjust_data.frame(x)
  # use same feature set as training
  x <- x[,object$x, drop = FALSE]

  prediction <- predict(object$model, x, probability = TRUE)
  prediction <- attr(prediction, "probabilities")
  prediction <- prediction[,object$slevels]
  prediction <- as.data.frame(prediction)
  colnames(prediction) <- object$slevels

  return(prediction)
}
