#'@title ELM
#'@description Creates a time series prediction object that
#' uses the Extreme Learning Machine (ELM).
#' It wraps the elmNNRcpp library.
#'@param preprocess normalization
#'@param input_size input size for machine learning model
#'@param nhid ensemble size
#'@param actfun defines the type to use, possible values: 'sig',
#' 'radbas', 'tribas', 'relu', 'purelin' (default).
#'@return returns a `ts_elm` object.
#'@examples
#'data(sin_data)
#'ts <- ts_data(sin_data$y, 10)
#'ts_head(ts, 3)
#'
#'samp <- ts_sample(ts, test_size = 5)
#'io_train <- ts_projection(samp$train)
#'io_test <- ts_projection(samp$test)
#'
#'model <- ts_elm(ts_norm_gminmax(), input_size=4, nhid=3, actfun="purelin")
#'model <- fit(model, x=io_train$input, y=io_train$output)
#'
#'prediction <- predict(model, x=io_test$input[1,], steps_ahead=5)
#'prediction <- as.vector(prediction)
#'output <- as.vector(io_test$output)
#'
#'ev_test <- evaluate(model, output, prediction)
#'ev_test
#'@export
ts_elm <- function(preprocess=NA, input_size=NA, nhid=NA, actfun='purelin') {
  obj <- ts_regsw(preprocess, input_size)
  if (is.na(nhid))
    nhid <- input_size/3
  obj$nhid <- nhid
  obj$actfun <- as.character(actfun)

  class(obj) <- append("ts_elm", class(obj))
  return(obj)
}

#'@import elmNNRcpp
#'@exportS3Method do_fit ts_elm
do_fit.ts_elm <- function(obj, x, y) {
  obj$model <- elmNNRcpp::elm_train(x, y, nhid = obj$nhid, actfun = obj$actfun, init_weights = "uniform_positive", bias = FALSE, verbose = FALSE)
  return(obj)
}

#'@import elmNNRcpp
#'@exportS3Method do_predict ts_elm
do_predict.ts_elm <- function(obj, x) {
  if (is.data.frame(x))
    x <- as.matrix(x)
  prediction <- elmNNRcpp::elm_predict(obj$model, x)
  return(prediction)
}
