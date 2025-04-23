#'@title Random Forest
#'@description Creates a time series prediction object that
#' uses the Random Forest.
#' It wraps the randomForest library.
#'@param preprocess normalization
#'@param input_size input size for machine learning model
#'@param nodesize node size
#'@param ntree number of trees
#'@param mtry number of attributes to build tree
#'@return returns a `ts_rf` object.
#'@examples
#'data(sin_data)
#'ts <- ts_data(sin_data$y, 10)
#'ts_head(ts, 3)
#'
#'samp <- ts_sample(ts, test_size = 5)
#'io_train <- ts_projection(samp$train)
#'io_test <- ts_projection(samp$test)
#'
#'model <- ts_rf(ts_norm_gminmax(), input_size=4, nodesize=3, ntree=50)
#'model <- fit(model, x=io_train$input, y=io_train$output)
#'
#'prediction <- predict(model, x=io_test$input[1,], steps_ahead=5)
#'prediction <- as.vector(prediction)
#'output <- as.vector(io_test$output)
#'
#'ev_test <- evaluate(model, output, prediction)
#'ev_test
#'@export
ts_rf <- function(preprocess=NA, input_size=NA, nodesize = 1, ntree = 10, mtry = NULL) {
  obj <- ts_regsw(preprocess, input_size)

  obj$nodesize <- nodesize
  obj$ntree <- ntree
  obj$mtry <- mtry

  class(obj) <- append("ts_rf", class(obj))
  return(obj)
}


#'@importFrom randomForest randomForest
#'@exportS3Method do_fit ts_rf
do_fit.ts_rf <- function(obj, x, y) {
  if (is.null(obj$mtry))
    obj$mtry <- ceiling(obj$input_size/3)
  obj$model <- randomForest::randomForest(x = as.data.frame(x), y = as.vector(y), mtry=obj$mtry, nodesize = obj$nodesize, ntree=obj$ntree)
  return(obj)
}

#'@importFrom stats predict
#'@exportS3Method do_predict ts_rf
do_predict.ts_rf <- function(obj, x) {
  prediction <- stats::predict(obj$model, as.data.frame(x))
  return(prediction)
}
