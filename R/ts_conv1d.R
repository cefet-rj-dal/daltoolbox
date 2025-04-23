#'@title Conv1D
#'@description Creates a time series prediction object that uses the Conv1D.
#' It wraps the pytorch library.
#'@param preprocess normalization
#'@param input_size input size for machine learning model
#'@param epochs maximum number of epochs
#'@return returns a `ts_conv1d` object.
#'@examples
#'#See an example of using `ts_conv1d` at this
#'#https://github.com/cefet-rj-dal/daltoolbox/blob/main/timeseries/ts_conv1d.md
#'@import reticulate
#'@export
ts_conv1d <- function(preprocess = NA, input_size = NA, epochs = 10000L) {
  obj <- ts_regsw(preprocess, input_size)
  obj$channels <- 1
  obj$epochs <- epochs
  class(obj) <- append("ts_conv1d", class(obj))
  return(obj)
}

#'@exportS3Method do_fit ts_conv1d
do_fit.ts_conv1d <- function(obj, x, y) {
  reticulate::source_python(system.file("python", "ts_conv1d.py", package = "daltoolbox"))

  if (is.null(obj$model))
    obj$model <- ts_conv1d_create(obj$channels, obj$input_size)

  df_train <- as.data.frame(x)
  df_train$t0 <- as.vector(y)

  obj$model <- ts_conv1d_fit(obj$model, df_train, obj$epochs, 0.001)

  return(obj)
}

#'@exportS3Method do_predict ts_conv1d
do_predict.ts_conv1d <- function(obj, x) {
  reticulate::source_python(system.file("python", "ts_conv1d.py", package = "daltoolbox"))

  X_values <- as.data.frame(x)
  X_values$t0 <- 0

  n <- nrow(X_values)
  prediction <- ts_conv1d_predict(obj$model, X_values)
  return(prediction)
}
