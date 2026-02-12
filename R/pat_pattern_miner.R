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

# Infer a lightweight schema signature without keeping the dataset in memory.
pattern_schema <- function(data) {
  if (inherits(data, "transactions")) {
    return(sort(arules::itemLabels(data)))
  }
  if (is.data.frame(data) || is.matrix(data)) {
    return(sort(colnames(data)))
  }
  stop("pattern_miner: unsupported data type for schema inference.")
}

validate_pattern_schema <- function(obj, data) {
  if (is.null(obj$schema)) {
    return(invisible(TRUE))
  }
  current <- pattern_schema(data)
  if (!identical(obj$schema, current)) {
    stop("pattern_miner: discover data schema differs from fit schema.")
  }
  invisible(TRUE)
}

#'@exportS3Method fit pattern_miner
fit.pattern_miner <- function(obj, data, ...) {
  obj$schema <- pattern_schema(data)
  return(obj)
}
