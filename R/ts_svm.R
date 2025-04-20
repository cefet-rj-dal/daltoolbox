#'@title SVM
#'@description Creates a time series prediction object that
#' uses the Support Vector Machine (SVM).
#' It wraps the e1071 library.
#'@param preprocess normalization
#'@param input_size input size for machine learning model
#'@param kernel SVM kernel (linear, radial, polynomial, sigmoid)
#'@param epsilon error threshold
#'@param cost this parameter controls the trade-off between achieving a low error on the training data and minimizing the model complexity
#'@return returns a `ts_svm` object.
#'@examples
#'data(sin_data)
#'ts <- ts_data(sin_data$y, 10)
#'ts_head(ts, 3)
#'
#'samp <- ts_sample(ts, test_size = 5)
#'io_train <- ts_projection(samp$train)
#'io_test <- ts_projection(samp$test)
#'
#'model <- ts_svm(ts_norm_gminmax(), input_size=4)
#'model <- fit(model, x=io_train$input, y=io_train$output)
#'
#'prediction <- predict(model, x=io_test$input[1,], steps_ahead=5)
#'prediction <- as.vector(prediction)
#'output <- as.vector(io_test$output)
#'
#'ev_test <- evaluate(model, output, prediction)
#'ev_test
#'@export
ts_svm <- function(preprocess=NA, input_size=NA, kernel="radial", epsilon=0, cost=10) {
  obj <- ts_regsw(preprocess, input_size)

  obj$kernel <- kernel #c("radial", "poly", "linear", "sigmoid")
  obj$epsilon <- epsilon #seq(0, 1, 0.1)
  obj$cost <- cost #=seq(10, 100, 10)

  class(obj) <- append("ts_svm", class(obj))
  return(obj)
}

#'@import e1071
#'@export
do_fit.ts_svm <- function(obj, x, y) {
  obj$model <- e1071::svm(x = as.data.frame(x), y = y, epsilon=obj$epsilon, cost=obj$cost, kernel=obj$kernel)
  return(obj)
}

#'@importFrom stats predict
#'@export
do_predict.ts_svm <- function(obj, x) {
  prediction <- stats::predict(obj$model, as.data.frame(x))
  return(prediction)
}
