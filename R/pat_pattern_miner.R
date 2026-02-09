#'@title Pattern miner
#'@description Base class for frequent pattern and sequence mining.
#'@return returns a `pattern_miner` object
#'@export
pattern_miner <- function() {
  obj <- dal_learner()
  class(obj) <- append("pattern_miner", class(obj))
  return(obj)
}

#'@exportS3Method action pattern_miner
action.pattern_miner <- function(obj, ...) {
  thiscall <- match.call(expand.dots = TRUE)
  thiscall[[1]] <- as.name("discover")
  result <- eval.parent(thiscall)
  return(result)
}

#'@title Discover
#'@description Generic for pattern discovery.
#'@param obj a `pattern_miner` object
#'@param ... optional arguments
#'@return discovered patterns
#'@export
discover <- function(obj, ...) {
  UseMethod("discover")
}

#'@exportS3Method discover default
discover.default <- function(obj, ...) {
  return(list())
}

#'@exportS3Method fit pattern_miner
fit.pattern_miner <- function(obj, data, ...) {
  obj$data <- data
  return(obj)
}
