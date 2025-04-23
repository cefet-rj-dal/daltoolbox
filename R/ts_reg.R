#'@title TSReg
#'@description Time Series Regression directly from time series
#'Ancestral class for non-sliding windows implementation.
#'@return returns `ts_reg` object
#'@examples
#'#See ?ts_arima for an example using Auto-regressive Integrated Moving Average
#'@export
ts_reg <- function() {
  obj <- predictor()
  class(obj) <- append("ts_reg", class(obj))
  return(obj)
}

#'@exportS3Method action ts_reg
action.ts_reg <- function(obj, ...) {
  thiscall <- match.call(expand.dots = TRUE)
  thiscall[[1]] <- as.name("predict")
  result <- eval.parent(thiscall)
  return(result)
}

#'@export
#'@exportS3Method predict ts_reg
predict.ts_reg <- function(object, x, ...) {
  return(x[,ncol(x)])
}

#'@title Fit Time Series Model
#'@description The actual time series model fitting.
#'This method should be override by descendants.
#'@param obj an object representing the model or algorithm to be fitted
#'@param x a matrix or data.frame containing the input features for training the model
#'@param y a vector or matrix containing the output values to be predicted by the model
#'@return returns a fitted object
#'@export
do_fit <- function(obj, x, y = NULL) {
  UseMethod("do_fit")
}

#'@title Predict Time Series Model
#'@description The actual time series model prediction.
#'This method should be override by descendants.
#'@param obj an object representing the fitted model or algorithm
#'@param x a matrix or data.frame containing the input features for making predictions
#'@return returns the predicted values
#'@export
do_predict <- function(obj, x) {
  UseMethod("do_predict")
}

#'@title MSE
#'@description Compute the mean squared error (MSE) between actual values and forecasts of a time series
#'@param actual real observations
#'@param prediction predicted observations
#'@return returns a number, which is the calculated MSE
#'@export
MSE.ts <- function (actual, prediction) {
  if (length(actual) != length(prediction))
    stop("actual and prediction have different lengths")
  n <- length(actual)
  res <- mean((actual - prediction)^2)
  res
}

#'@title sMAPE
#'@description Compute the symmetric mean absolute percent error (sMAPE)
#'@param actual real observations
#'@param prediction predicted observations
#'@return returns the sMAPE between the actual and prediction vectors
#'@export
sMAPE.ts <- function (actual, prediction) {
  if (length(actual) != length(prediction))
    stop("actual and prediction have different lengths")
  n <- length(actual)
  num <- abs(actual - prediction)
  denom <- (abs(actual) + abs(prediction))/2
  i <- denom != 0
  num <- num[i]
  denom <- denom[i]
  res <- (1/n) * sum(num/denom)
  res
}

#'@title R2
#'@description Compute the R-squared (R2) between actual values and forecasts of a time series
#'@param actual real observations
#'@param prediction predicted observations
#'@return returns a number, which is the calculated R2
#'@export
R2.ts <- function (actual, prediction) {
  if (length(actual) != length(prediction))
    stop("actual and prediction have different lengths")
  res <-  1 - sum((prediction - actual)^2)/sum((mean(actual) - actual)^2)
  res
}


#'@exportS3Method evaluate ts_reg
evaluate.ts_reg <- function(obj, values, prediction, ...) {
  result <- list(values=values, prediction=prediction)

  result$smape <- sMAPE.ts(values, prediction)
  result$mse <- MSE.ts(values, prediction)
  result$R2 <- R2.ts(values, prediction)

  result$metrics <- data.frame(mse=result$mse, smape=result$smape, R2 = result$R2)

  return(result)
}

