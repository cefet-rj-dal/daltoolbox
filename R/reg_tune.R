#'@title Regression Tune
#'@description Creates an object for tuning regression models
#'@param base_model base model for tuning
#'@param folds number of folds for cross-validation
#'@param ranges a list of hyperparameter ranges to explore
#'@return returns a `reg_tune` object.
#'@examples
#'# preparing dataset for random sampling
#'data(Boston)
#'sr <- sample_random()
#'sr <- train_test(sr, Boston)
#'train <- sr$train
#'test <- sr$test
#'
#'# hyper parameter setup
#'tune <- reg_tune(reg_mlp("medv"), ranges = list(size=c(3), decay=c(0.1,0.5)))
#'
#'# hyper parameter optimization
#'model <- fit(tune, train, ranges)
#'
#'test_prediction <- predict(model, test)
#'test_predictand <- test[,"medv"]
#'test_eval <- evaluate(model, test_predictand, test_prediction)
#'test_eval$metrics
#'@export
reg_tune <- function(base_model, folds = 10, ranges = NULL) {
  obj <- dal_tune(base_model, folds, ranges)
  obj$name <- ""
  class(obj) <- append("reg_tune", class(obj))
  return(obj)
}


#'@importFrom stats predict
#'@export
#'@exportS3Method fit reg_tune
fit.reg_tune <- function(obj, data, ...) {

  build_model <- function(obj, ranges, data) {
    model <- obj$base_model
    model <- set_params(model, ranges)
    model <- fit(model, data)
    return(model)
  }

  prepare_ranges <- function(obj, ranges) {
    ranges <- expand.grid(ranges)
    ranges$key <- 1:nrow(ranges)
    obj$ranges <- ranges
    return(obj)
  }

  evaluate_error <- function(model, data) {
    x <- as.matrix(data[,model$x])
    y <- data[,model$attribute]
    prediction <- as.vector(stats::predict(model, x))
    error <- evaluate(model, y, prediction)$mse
    return(error)
  }

  ranges <- obj$ranges

  obj <- prepare_ranges(obj, ranges)
  ranges <- obj$ranges

  n <- nrow(ranges)
  i <- 1
  hyperparameters <- NULL
  if (n > 1) {
    ref <- data.frame(i = 1:nrow(data), idx = 1:nrow(data))
    folds <- k_fold(sample_random(), ref, obj$folds)
    nfolds <- length(folds)
    for (j in 1:nfolds) {
      tt <- train_test_from_folds(folds, j)
      error <- rep(0, n)
      msg <- rep("", n)
      for (i in 1:n) {
        err <- tryCatch(
          {
            model <- build_model(obj, ranges[i,], data[tt$train$i,])
            error[i] <- evaluate_error(model, data[tt$test$i,])
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

  model <- build_model(obj, ranges[i,], data)
  if (n == 1) {
    error <- evaluate_error(model, data)
    hyperparameters <- cbind(ranges, error)
  }
  attr(model, "params") <- as.list(ranges[i,])
  attr(model, "hyperparameters") <- hyperparameters

  return(model)
}


#'@importFrom dplyr filter summarise group_by
#'@exportS3Method select_hyper reg_tune
select_hyper.reg_tune <- function(obj, hyperparameters) {
  msg <- error <- 0
  hyper_summary <- hyperparameters |> dplyr::filter(msg == "") |>
    dplyr::group_by(key) |> dplyr::summarise(error = mean(error, na.rm=TRUE))

  mim_error <- hyper_summary |> dplyr::summarise(error = min(error))

  key <- which(hyper_summary$error == mim_error$error)
  i <- min(hyper_summary$key[key])
  return(i)
}
