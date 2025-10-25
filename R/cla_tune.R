#'@title Classification Tune
#'@description This function performs a grid search or random search over specified hyperparameter values to optimize a base classification model
#'@param base_model base model for tuning
#'@param folds number of folds for cross-validation
#'@param ranges a list of hyperparameter ranges to explore
#'@param metric metric used to optimize
#'@return returns a `cla_tune` object
#'@examples
#'# preparing dataset for random sampling
#'sr <- sample_random()
#'sr <- train_test(sr, iris)
#'train <- sr$train
#'test <- sr$test
#'
#'# hyper parameter setup
#'tune <- cla_tune(cla_mlp("Species", levels(iris$Species)),
#'   ranges=list(size=c(3:5), decay=c(0.1)))
#'
#'# hyper parameter optimization
#'model <- fit(tune, train)
#'
#'# testing optimization
#'test_prediction <- predict(model, test)
#'test_predictand <- adjust_class_label(test[,"Species"])
#'test_eval <- evaluate(model, test_predictand, test_prediction)
#'test_eval$metrics
#'@export
cla_tune <- function(base_model, folds=10, ranges=NULL, metric="accuracy") {
  obj <- dal_tune(base_model, folds, ranges)
  obj$name <- ""
  obj$metric <- metric
  class(obj) <- append("cla_tune", class(obj))
  return(obj)
}


#'@title tune hyperparameters of ml model
#'@description Tunes the hyperparameters of a machine learning model for classification
#'@param obj an object containing the model and tuning configuration
#'@param data the dataset used for training and evaluation
#'@param ... optional arguments
#'@return a fitted obj
#'@importFrom stats predict
#'@export
#'@exportS3Method fit cla_tune
fit.cla_tune <- function(obj, data, ...) {

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

  evaluate_metric <- function(model, data) {
    # predict on same feature set; metric computed via evaluate()
    x <- as.matrix(data[,model$x])
    y <- adjust_class_label(data[,model$attribute])
    prediction <- stats::predict(model, x)
    metric <- evaluate(model, y, prediction)$metrics[1,obj$metric]
    return(metric)
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
      metric <- rep(0, n)
      msg <- rep("", n)
      for (i in 1:n) {
        err <- tryCatch(
          {
            model <- build_model(obj, ranges[i,], data[tt$train$i,])
            metric[i] <- evaluate_metric(model, data[tt$test$i,])
            ""
          },
          error = function(cond) {
            err <- sprintf("tune: %s", as.character(cond))
          }
        )
      }
      hyperparameters <- rbind(hyperparameters, cbind(ranges, metric, msg))
    }
    hyperparameters$error[hyperparameters$msg != ""] <- NA
    i <- select_hyper(obj, hyperparameters)
  }

  model <- build_model(obj, ranges[i,], data)
  if (n == 1) {
    metric <- evaluate_metric(model, data)
    hyperparameters <- cbind(ranges, metric)
  }
  attr(model, "params") <- as.list(ranges[i,])
  attr(model, "hyperparameters") <- hyperparameters

  return(model)
}


#'@title selection of hyperparameters
#'@description Selects the optimal hyperparameter by maximizing the average classification metric. It wraps dplyr library.
#'@param obj an object representing the model or tuning process
#'@param hyperparameters a dataframe with columns `key` (hyperparameter configuration) and `metric` (classification metric)
#'@return returns a optimized key number of hyperparameters
#'@importFrom dplyr filter summarise group_by
#'@exportS3Method select_hyper cla_tune
select_hyper.cla_tune <- function(obj, hyperparameters) {
  msg <- metric <- 0
  hyper_summary <- hyperparameters |> dplyr::filter(msg == "") |>
    dplyr::group_by(key) |> dplyr::summarise(metric = mean(metric, na.rm=TRUE))

  max_metric <- hyper_summary |> dplyr::summarise(metric = max(metric))

  key <- which(hyper_summary$metric == max_metric$metric)
  # tie-breaker: choose smallest key among maxima
  i <- min(hyper_summary$key[key])
  return(i)
}


