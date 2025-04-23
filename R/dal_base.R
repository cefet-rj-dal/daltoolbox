#'@title Class dal_base
#'@description The dal_base class is an abstract class for all dal descendants classes. It provides both fit() and action() functions
#'@return returns a dal_base object
#'@examples trans <- dal_base()
#'@export
dal_base <- function() {
  obj <- list()
  attr(obj, "class") <- "dal_base"
  return(obj)
}

#'@title Fit
#'@description Applies the `fit` method to a model object to train or configure it using the provided data and optional arguments
#'@param obj object
#'@param ... optional arguments.
#'@return returns a object after fitting
#'@examples
#'data(iris)
#'# an example is minmax normalization
#'trans <- minmax()
#'trans <- fit(trans, iris)
#'tiris <- action(trans, iris)
#'@export
fit <- function(obj, ...) {
  UseMethod("fit")
}

#'@exportS3Method fit default
fit.default <- function(obj, ...) {
  return(obj)
}

#'@title Action
#'@description Executes the action of model applied in provided data
#'@param obj object: a dal_base object to apply the transformation on the input dataset.
#'@param ... optional arguments.
#'@return returns the result of an action of the model applied in provided data
#'@examples
#'data(iris)
#'# an example is minmax normalization
#'trans <- minmax()
#'trans <- fit(trans, iris)
#'tiris <- action(trans, iris)
#'@export
action <- function(obj, ...) {
  UseMethod("action")
}

#'@exportS3Method action default
action.default <- function(obj, ...) {
  par <- c(as.list(environment()), list(...))
  data <- NULL
  if (length(par) > 1)
    data <- par[[2]]
  return (data)
}

#'@title Assign parameters
#'@description set_params function assigns all parameters to the attributes presented in the object.
#'@param obj object of class dal_base
#'@param params parameters to set obj
#'@return returns an object with parameters set
#'@examples
#'obj <- set_params(dal_base(), list(x = 0))
#'@export
set_params <- function(obj, params) {
  UseMethod("set_params")
}

#'@title Default Assign parameters
#'@description Default method for `set_params` which returns the object unchanged
#'@param obj object
#'@param params parameters
#'@return returns the object unchanged
#'@export
#'@exportS3Method set_params default
set_params.default <- function(obj, params) {
  return(obj)
}

#'@exportS3Method set_params dal_base
set_params.dal_base <- function(obj, params) {
  if (!is.null(params)) {
    params <- as.list(params)
    nobj <- names(obj)
    nobj <- nobj[nobj != ""]
    nparams <- names(params)
    nparams <- nparams[nparams != ""]

    for (i in 1:length(nparams)) {
      j <- which(nparams[i] == nobj)
      if (length(j)> 0) {
        val <- params[[nparams[i]]]
        if (is.factor(val))
          val <- as.character(val)
        obj[[nobj[j]]] <- val
      }
    }
  }
  return(obj)
}
