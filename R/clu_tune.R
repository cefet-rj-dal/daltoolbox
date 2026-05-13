#'@title Clustering tuning (intrinsic metric)
#'@description Tune clustering hyperparameters by evaluating an intrinsic metric over a parameter grid and selecting the elbow (max curvature).
#'@param base_model base model for tuning
#'@param folds number of folds for cross-validation
#'@param ranges a list of hyperparameter ranges to explore
#'@return returns a `clu_tune` object.
#'@references
#' Satopaa, V. et al. (2011). Finding a “Kneedle” in a Haystack.
#'@examples
#'data(iris)
#'
#'# fit model
#'model <- clu_tune(cluster_kmeans(k = 2), ranges = list(k = 2:10))
#'
#'model <- fit(model, iris[,1:4])
#'model$k
#'@export
clu_tune <- function(base_model, folds=10, ranges=NULL) {
  obj <- dal_tune(base_model, folds, ranges)
  utils <- if (!is.null(base_model$clu_utils)) base_model$clu_utils else cluutils()
  obj$base_model <- base_model
  obj$metric <- base_model$metric
  obj$selector <- base_model$selector
  obj$clu_utils <- utils
  obj$name <- ""
  class(obj) <- append("clu_tune", class(obj))
  return(obj)
}

#'@importFrom stats predict
#'@export
#'@exportS3Method fit clu_tune
fit.clu_tune <- function(obj, data, ...) {

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

  ranges <- obj$ranges

  obj <- prepare_ranges(obj, ranges)
  ranges <- obj$ranges
  hyperparameters <- ranges
  hyperparameters$metric <- NA
  hyperparameters$goal <- NA_character_
  hyperparameters$msg <- ""

  n <- nrow(ranges)
  i <- 1

  for (i in 1:n) {
    err <- tryCatch(
      {
        model <- build_model(obj, ranges[i,], data)
        clu <- cluster(model, model$train_data)
        metric_res <- obj$metric(data = model$train_data, cluster = clu, obj = model)
        hyperparameters$metric[i] <- metric_res$value
        hyperparameters$goal[i] <- metric_res$goal
        ""
      },
      error = function(cond) {
        sprintf("tune: %s", as.character(cond))
      }
    )
    if (err != "") {
      hyperparameters$msg[i] <- err
      hyperparameters$metric[i] <- NA_real_
    }
  }

  valid <- hyperparameters$msg == ""
  if (sum(valid) > 1) {
    goal <- hyperparameters$goal[which(valid)[1]]
    selected <- obj$selector(hyperparameters$metric, goal = goal)
    if (!is.na(selected)) {
      i <- selected
    }
  } else if (sum(valid) == 1) {
    i <- which(valid)[1]
  }

  model <- build_model(obj, ranges[i, ], data)
  attr(model, "params") <- as.list(ranges[i, ])
  attr(model, "hyperparameters") <- hyperparameters
  return(model)
}

