#'@title LSTM
#'@description Creates a time series prediction object that uses the LSTM.
#' It wraps the pytorch library.
#'@param preprocess normalization
#'@param input_size input size for machine learning model
#'@param epochs maximum number of epochs
#'@return returns a `ts_lstm` object.
#'@examples
#'#See an example of using `ts_ts_lstmconv1d` at this
#'#https://github.com/cefet-rj-dal/daltoolbox/blob/main/timeseries/ts_lstm.md
#'@import reticulate
#'@export
ts_lstm <- function(preprocess = NA, input_size = NA, epochs = 10000L) {
  obj <- ts_regsw(preprocess, input_size)
  obj$epochs <- epochs
  class(obj) <- append("ts_lstm", class(obj))

  return(obj)
}

#'@exportS3Method do_fit ts_lstm
do_fit.ts_lstm <- function(obj, x, y) {
  if (!exists("ts_lstm_create"))
    reticulate::source_python(system.file("python", "ts_lstm.py", package = "daltoolbox"))

  if (is.null(obj$model))
    obj$model <- ts_lstm_create(obj$input_size, obj$input_size)

  df_train <- as.data.frame(x)
  df_train$t0 <- as.vector(y)

  obj$model <- ts_lstm_fit(obj$model, df_train, obj$epochs, 0.001)

  return(obj)
}

#'@exportS3Method do_predict ts_lstm
do_predict.ts_lstm <- function(obj, x) {
  if (!exists("ts_lstm_predict"))
    reticulate::source_python(system.file("python", "ts_lstm.py", package = "daltoolbox"))

  X_values <- as.data.frame(x)
  X_values$t0 <- 0

  n <- nrow(X_values)
  prediction <- ts_lstm_predict(obj$model, X_values)
  return(prediction)
}
