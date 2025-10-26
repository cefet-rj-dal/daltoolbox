#'@title DAL Transform
#'@description Base class for data transformations with optional `fit()`/`inverse_transform()` support.
#'@details The default `transform()` calls the underlying `action.default()`; subclasses should implement
#' `transform.className` and optionally `inverse_transform.className`.
#'@return returns a `dal_transform` object
#'@examples
#'# See ?minmax or ?zscore for examples
#'@export
dal_transform <- function() {
  obj <- dal_base()
  class(obj) <- append("dal_transform", class(obj))
  return(obj)
}

#'@title Transform
#'@description Generic to apply a transformation to data.
#'@param obj a `dal_transform` object.
#'@param ... optional arguments.
#'@return returns a transformed data.
#'@examples
#'#See ?minmax for an example of transformation
#'@export
transform <- function(obj, ...) {
  UseMethod("transform")
}

#'@exportS3Method transform default
transform.default <- function(obj, ...) {
  thiscall <- match.call(expand.dots = TRUE)
  # proxy transform to the default action implementation
  thiscall[[1]] <- as.name("action.default")
  result <- eval.parent(thiscall)
  return (result)
}

#'@title Action implementation for transform
#'@description Default `action()` implementation that proxies to `transform()` for transforms.
#'@param obj object
#'@param ... optional arguments
#'@return returns a transformed data
#'@examples
#'#See ?minmax for an example of transformation
#'@exportS3Method action dal_transform
action.dal_transform <- function(obj, ...) {
  thiscall <- match.call(expand.dots = TRUE)
  # ensure action() on a transform dispatches to transform()
  thiscall[[1]] <- as.name("transform")
  result <- eval.parent(thiscall)
  return(result)
}

#'@title Inverse Transform
#'@description Optional inverse operation for a transformation; defaults to identity.
#'@param obj a dal_transform object.
#'@param ... optional arguments.
#'@return dataset inverse transformed.
#'@examples
#'#See ?minmax for an example of transformation
#'@export
inverse_transform <- function(obj, ...) {
  UseMethod("inverse_transform")
}

#'@exportS3Method inverse_transform default
inverse_transform.default <- function(obj, ...) {
  thiscall <- match.call(expand.dots = TRUE)
  # by default, inverse_transform behaves like a no-op action
  thiscall[[1]] <- as.name("action.default")
  result <- eval.parent(thiscall)
  return (result)
}
