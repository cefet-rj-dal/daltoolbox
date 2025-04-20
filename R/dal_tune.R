#'@title DAL Tune
#'@description Creates an ancestor class for hyperparameter optimization,
#'allowing the tuning of a base model using cross-validation.
#'@param base_model base model for tuning
#'@param folds number of folds for cross-validation
#'@return returns a `dal_tune` object
#'@examples
#'#See ?cla_tune for classification tuning
#'#See ?reg_tune for regression tuning
#'#See ?ts_tune for time series tuning
#'@export
dal_tune <- function(base_model, folds=10) {
  obj <- dal_base()
  obj$base_model <- base_model
  obj$folds <- folds
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

#'@export
select_hyper.default <- function(obj, hyperparameters) {
  return(length(hyperparameters))
}

