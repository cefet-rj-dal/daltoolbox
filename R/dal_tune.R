#'@title DAL Tune
#'@description Creates an ancestor class for hyperparameter optimization,
#'allowing the tuning of a base model using cross-validation.
#'@param base_model base model for tuning
#'@param folds number of folds for cross-validation
#'@param ranges a list of hyperparameter ranges to explore
#'@return returns a `dal_tune` object
#'@examples
#'#See ?cla_tune for classification tuning
#'#See ?reg_tune for regression tuning
#'#See ?ts_tune for time series tuning
#'@export
dal_tune <- function(base_model, folds=10, ranges) {
  obj <- dal_base()
  obj$base_model <- base_model
  obj$folds <- folds
  obj$ranges <- ranges
  class(obj) <- append("dal_tune", class(obj))
  return(obj)
}

#'@title Selection hyper parameters
#'@description Selects the optimal hyperparameters from a dataset resulting from k-fold cross-validation
#'@param obj the object or model used for hyperparameter selection.
#'@param hyperparameters data set with hyper parameters and quality measure from execution
#'@return returns the index of selected hyper parameter
#'@export
select_hyper <- function(obj, hyperparameters) {
  UseMethod("select_hyper")
}

#'@exportS3Method select_hyper default
select_hyper.default <- function(obj, hyperparameters) {
  # default: choose last row (useful when ranges has a single configuration)
  return(length(hyperparameters))
}

