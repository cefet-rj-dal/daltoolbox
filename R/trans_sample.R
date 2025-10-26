#'@title Data sampling abstractions
#'@description Base class for sampling strategies that provide train/test splitting and k‑fold partitioning.
#' Two standard implementations are `sample_random()` and `sample_stratified()`.
#'@return returns an object of class `data_sample`
#'@examples
#'#using random sampling
#'sample <- sample_random()
#'tt <- train_test(sample, iris)
#'
#'# distribution of train
#'table(tt$train$Species)
#'
#'# preparing dataset into four folds
#'folds <- k_fold(sample, iris, 4)
#'
#'# distribution of folds
#'tbl <- NULL
#'for (f in folds) {
#'  tbl <- rbind(tbl, table(f$Species))
#'}
#'head(tbl)
#'@export
data_sample <- function() {
  obj <- dal_transform()
  class(obj) <- append("data_sample", class(obj))
  return(obj)
}

#'@title Train-Test Partition
#'@description Partition a dataset into training and test sets using a sampling strategy.
#'@param obj an object of a class that supports the `train_test` method
#'@param data dataset to be partitioned
#'@param perc a numeric value between 0 and 1 specifying the proportion of data to be used for training
#'@param ... additional optional arguments passed to specific methods.
#'@return returns an list with two elements:
#' \itemize{
#'   \item train: A data frame containing the training set
#'   \item test: A data frame containing the test set
#' }
#'@examples
#'#using random sampling
#'sample <- sample_random()
#'tt <- train_test(sample, iris)
#'
#'# distribution of train
#'table(tt$train$Species)
#'@export
train_test <- function(obj, data, perc=0.8, ...) {
  UseMethod("train_test")
}

#'@exportS3Method train_test default
train_test.default <- function(obj, data, perc=0.8, ...) {
  return(list())
}

#'@title K-fold sampling
#'@description Split a dataset into `k` folds using a sampling strategy.
#'@param obj an object representing the sampling method
#'@param data dataset to be partitioned
#'@param k number of folds
#'@return returns a list of `k` data frames
#'@examples
#'#using random sampling
#'sample <- sample_random()
#'
#'# preparing dataset into four folds
#'folds <- k_fold(sample, iris, 4)
#'
#'# distribution of folds
#'tbl <- NULL
#'for (f in folds) {
#'  tbl <- rbind(tbl, table(f$Species))
#'}
#'head(tbl)
#'@export
k_fold <- function(obj, data, k) {
  UseMethod("k_fold")
}

#'@exportS3Method k_fold default
k_fold.default <- function(obj, data, k) {
  return(list())
}


