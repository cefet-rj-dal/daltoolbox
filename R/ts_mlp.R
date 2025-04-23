#'@title MLP
#'@description Creates a time series prediction object that
#' uses the Multilayer Perceptron (MLP).
#' It wraps the nnet library.
#'@param preprocess normalization
#'@param input_size input size for machine learning model
#'@param size number of neurons inside hidden layer
#'@param decay decay parameter for MLP
#'@param maxit maximum number of iterations
#'@return returns a `ts_mlp` object.
#'@examples
#'data(sin_data)
#'ts <- ts_data(sin_data$y, 10)
#'ts_head(ts, 3)
#'
#'samp <- ts_sample(ts, test_size = 5)
#'io_train <- ts_projection(samp$train)
#'io_test <- ts_projection(samp$test)
#'
#'model <- ts_mlp(ts_norm_gminmax(), input_size=4, size=4, decay=0)
#'model <- fit(model, x=io_train$input, y=io_train$output)
#'
#'prediction <- predict(model, x=io_test$input[1,], steps_ahead=5)
#'prediction <- as.vector(prediction)
#'output <- as.vector(io_test$output)
#'
#'ev_test <- evaluate(model, output, prediction)
#'ev_test
#'@export
ts_mlp <- function(preprocess=NA, input_size=NA, size=NA, decay=0.01, maxit=1000) {
  obj <- ts_regsw(preprocess, input_size)
  if (is.na(size))
    size <- ceiling(input_size/3)

  obj$size <- size
  obj$decay <- decay
  obj$maxit <- maxit

  class(obj) <- append("ts_mlp", class(obj))
  return(obj)
}


#'@import nnet
#'@exportS3Method do_fit ts_mlp
do_fit.ts_mlp <- function(obj, x, y) {
  obj$model <- nnet::nnet(x = x, y = y, size = obj$size, decay=obj$decay, maxit = obj$maxit, linout=TRUE, trace = FALSE)
  return(obj)
}
