# DAL Library
# version 2.1

#'@title Clustering Tune
#'@description Creates an object for tuning clustering models.
#'This object can be used to fit and optimize clustering algorithms by specifying hyperparameter ranges
#'@param base_model base model for tuning
#'@return returns a `clu_tune` object.
#'@examples
#'data(iris)
#'
#'# fit model
#'model <- clu_tune(cluster_kmeans(k = 0))
#'ranges <- list(k = 1:10)
#'model <- fit(model, iris[,1:4], ranges)
#'model$k
#'@export
clu_tune <- function(base_model) {
  obj <- dal_base()
  obj$base_model <- base_model
  obj$name <- ""
  class(obj) <- append("clu_tune", class(obj))
  return(obj)
}

#'@importFrom stats predict
#'@export
fit.clu_tune <- function(obj, data, ranges, ...) {

  build_cluster <- function(obj, ranges, data) {
    model <- obj$base_model
    model <- set_params(model, ranges)
    result <- cluster(model, data)
    return(result)
  }

  prepare_ranges <- function(obj, ranges) {
    ranges <- expand.grid(ranges)
    ranges$key <- 1:nrow(ranges)
    obj$ranges <- ranges
    return(obj)
  }

  obj <- prepare_ranges(obj, ranges)
  ranges <- obj$ranges
  ranges$metric <- NA

  n <- nrow(ranges)
  i <- 1
  if (n > 1) {
    msg <- rep("", n) #save later msg in hyper parameters
    for (i in 1:n) {
      err <- tryCatch(
        {
          clu <- build_cluster(obj, ranges[i,], data)
          ranges$metric[i] <- attr(clu, "metric")
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
    myfit <- fit_curvature_max()
    res <- transform(myfit, ranges$metric)
    i <- res$x
  }
  model <- set_params(obj$base_model, ranges[i,])
  return(model)
}

