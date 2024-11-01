#'@title KNN time series prediction
#'@description Creates a prediction object that
#' uses the K-Nearest Neighbors (KNN) method for time series regression
#'@param preprocess normalization
#'@param input_size input size for machine learning model
#'@param k number of k neighbors
#'@return returns a `ts_knn` object.
#'@examples
#'data(sin_data)
#'ts <- ts_data(sin_data$y, 10)
#'ts_head(ts, 3)
#'
#'samp <- ts_sample(ts, test_size = 5)
#'io_train <- ts_projection(samp$train)
#'io_test <- ts_projection(samp$test)
#'
#'model <- ts_knn(ts_norm_gminmax(), input_size=4, k=3)
#'model <- fit(model, x=io_train$input, y=io_train$output)
#'
#'prediction <- predict(model, x=io_test$input[1,], steps_ahead=5)
#'prediction <- as.vector(prediction)
#'output <- as.vector(io_test$output)
#'
#'ev_test <- evaluate(model, output, prediction)
#'ev_test
#'@export
ts_knn <- function(preprocess=NA, input_size=NA, k=NA) {
  obj <- ts_regsw(preprocess, input_size)
  if (is.na(k))
    k <- input_size/3
  obj$k <- k

  class(obj) <- append("ts_knn", class(obj))
  return(obj)
}

#'@importFrom FNN knn.reg
#'@export
do_fit.ts_knn <- function(obj, x, y) {
  x <- adjust_data.frame(x)
  y <- adjust_data.frame(y)

  obj$model <- list(x=x, y=y, k=obj$k)

  return(obj)
}

#'@importFrom FNN knn.reg
#'@export
do_predict.ts_knn <- function(obj, x) {
  #develop from FNN https://daviddalpiaz.github.io/r4sl/knn-reg.html
  x <- adjust_data.frame(x)
  prediction <- FNN::knn.reg(train = obj$model$x, test = x, y = obj$model$y, k = obj$model$k)
  prediction <- prediction$pred
  return(prediction)
}
