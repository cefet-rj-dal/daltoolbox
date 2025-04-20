#'@title Time Series Tune
#'@description Creates a `ts_tune` object for tuning hyperparameters of a time series model.
#'This function sets up a tuning process for the specified base model by exploring different
#'configurations of hyperparameters using cross-validation.
#'@param input_size input size for machine learning model
#'@param base_model base model for tuning
#'@param folds number of folds for cross-validation
#'@return returns a `ts_tune` object
#'@examples
#'data(sin_data)
#'ts <- ts_data(sin_data$y, 10)
#'ts_head(ts, 3)
#'
#'samp <- ts_sample(ts, test_size = 5)
#'io_train <- ts_projection(samp$train)
#'io_test <- ts_projection(samp$test)
#'
#'tune <- ts_tune(input_size=c(3:5), base_model = ts_elm(ts_norm_gminmax()))
#'ranges <- list(nhid = 1:5, actfun=c('purelin'))
#'
#'# Generic model tunning
#'model <- fit(tune, x=io_train$input, y=io_train$output, ranges)
#'
#'prediction <- predict(model, x=io_test$input[1,], steps_ahead=5)
#'prediction <- as.vector(prediction)
#'output <- as.vector(io_test$output)
#'
#'ev_test <- evaluate(model, output, prediction)
#'ev_test
#'@export
ts_tune <- function(input_size, base_model, folds=10) {
  obj <- dal_tune(base_model, folds)
  obj$input_size <- input_size
  obj$name <- ""
  class(obj) <- append("ts_tune", class(obj))
  return(obj)
}

#'@importFrom stats predict
#'@export
fit.ts_tune <- function(obj, x, y, ranges, ...) {

  build_model <- function(obj, ranges, x, y) {
    model <- obj$base_model
    model$input_size <- ranges$input_size
    model <- set_params(model, ranges)
    model <- fit(model, x, y)
    return(model)
  }

  prepare_ranges <- function(obj, ranges) {
    ranges <- append(list(input_size = obj$input_size), ranges)
    ranges <- expand.grid(ranges)
    ranges$key <- 1:nrow(ranges)
    obj$ranges <- ranges
    return(obj)
  }

  evaluate_error <- function(model, i, x, y) {
    x <- x[i,]
    y <- as.vector(y[i,])
    prediction <- as.vector(stats::predict(model, x))
    error <- evaluate(model, y, prediction)$mse
    return(error)
  }

  obj <- prepare_ranges(obj, ranges)
  ranges <- obj$ranges

  n <- nrow(ranges)
  i <- 1
  hyperparameters <- NULL
  if (n > 1) {
    data <- data.frame(i = 1:nrow(x), idx = 1:nrow(x))
    folds <- k_fold(sample_random(), data, obj$folds)
    nfolds <- length(folds)
    for (j in 1:nfolds) {
      tt <- train_test_from_folds(folds, j)
      error <- rep(0, n)
      msg <- rep("", n)
      for (i in 1:n) {
        err <- tryCatch(
          {
            model <- build_model(obj, ranges[i,], x[tt$train$i,], y[tt$train$i,])
            error[i] <- evaluate_error(model, tt$test$i, x, y)
            ""
          },
          error = function(cond) {
            err <- sprintf("tune: %s", as.character(cond))
          }
        )
        if (err != "") {
          msg[i] <- err
        }
      }
      hyperparameters <- rbind(hyperparameters, cbind(ranges, error, msg))
    }
    hyperparameters$error[hyperparameters$msg != ""] <- NA
    i <- select_hyper(obj, hyperparameters)
  }

  model <- build_model(obj, ranges[i,], x, y)
  if (n == 1) {
    prediction <- stats::predict(model, x)
    error <- evaluate(model, y, prediction)$mse
    hyperparameters <- cbind(ranges, error)
  }

  attr(model, "params") <- as.list(ranges[i,])
  attr(model, "hyperparameters") <- hyperparameters

  return(model)
}

#'@title Select Optimal Hyperparameters for Time Series Models
#'@description Identifies the optimal hyperparameters by minimizing the error from a dataset of hyperparameters.
#' The function selects the hyperparameter configuration that results in the lowest average error.
#' It wraps the dplyr library.
#'@param obj a `ts_tune` object containing the model and tuning settings
#'@param hyperparameters hyperparameters dataset
#'@return returns the optimized key number of hyperparameters
#'@importFrom dplyr filter summarise group_by
#'@export
select_hyper.ts_tune <- function(obj, hyperparameters) {
  msg <- error <- 0
  hyper_summary <- hyperparameters |> dplyr::filter(msg == "") |>
    dplyr::group_by(key) |> dplyr::summarise(error = mean(error, na.rm=TRUE))

  mim_error <- hyper_summary |> dplyr::summarise(error = min(error, na.rm=TRUE))

  key <- which(hyper_summary$error == mim_error$error)
  i <- min(hyper_summary$key[key])
  return(i)
}
