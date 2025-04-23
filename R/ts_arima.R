#'@title ARIMA
#'@description Creates a time series prediction object that
#' uses the AutoRegressive Integrated Moving Average (ARIMA).
#' It wraps the forecast library.
#'@return returns a `ts_arima` object.
#'@examples
#'data(sin_data)
#'ts <- ts_data(sin_data$y, 0)
#'ts_head(ts, 3)
#'
#'samp <- ts_sample(ts, test_size = 5)
#'io_train <- ts_projection(samp$train)
#'io_test <- ts_projection(samp$test)
#'
#'model <- ts_arima()
#'model <- fit(model, x=io_train$input, y=io_train$output)
#'
#'prediction <- predict(model, x=io_test$input[1,], steps_ahead=5)
#'prediction <- as.vector(prediction)
#'output <- as.vector(io_test$output)
#'
#'ev_test <- evaluate(model, output, prediction)
#'ev_test
#'@export
ts_arima <- function() {
  obj <- ts_reg()

  class(obj) <- append("ts_arima", class(obj))
  return(obj)
}

#'@importFrom forecast auto.arima
#'@exportS3Method fit ts_arima
fit.ts_arima <- function(obj, x, y = NULL, ...) {
  obj$model <- forecast::auto.arima(x, allowdrift = TRUE, allowmean = TRUE)
  order <- obj$model$arma[c(1, 6, 2, 3, 7, 4, 5)]
  obj$p <- order[1]
  obj$d <- order[2]
  obj$q <- order[3]
  obj$drift <- (NCOL(obj$model$xreg) == 1) && is.element("drift", names(obj$model$coef))
  params <- list(p = obj$p, d = obj$d, q = obj$q, drift = obj$drift)
  attr(obj, "params") <- params

  return(obj)
}

#'@importFrom forecast forecast
#'@importFrom forecast Arima
#'@importFrom forecast auto.arima
#'@exportS3Method predict ts_arima
predict.ts_arima <- function(object, x, y = NULL, steps_ahead=NULL, ...) {
  if (!is.null(x) && (length(object$model$x) == length(x)) && (sum(object$model$x-x) == 0)){
    #get adjusted data
    pred <- object$model$x - object$model$residuals
  }
  else {
    if (is.null(steps_ahead))
      steps_ahead <- length(x)
    if ((steps_ahead == 1) && (length(x) != 1)) {
      pred <- NULL
      model <- object$model
      i <- 1
      y <- model$x
      while (i <= length(x)) {
        pred <- c(pred, forecast::forecast(model, h = 1)$mean)
        y <- c(y, x[i])

        model <- tryCatch(
          {
            forecast::Arima(y, order=c(object$p, object$d, object$q), include.drift = object$drift)
          },
          error = function(cond) {
            forecast::auto.arima(y, allowdrift = TRUE, allowmean = TRUE)
          }
        )
        i <- i + 1
      }
    }
    else {
      pred <- forecast::forecast(object$model, h = steps_ahead)$mean
    }
  }
  return(pred)
}
