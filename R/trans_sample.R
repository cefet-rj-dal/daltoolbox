#'@title Data Sample
#'@description The data_sample function in R is used to randomly
#' sample data from a given data frame. It can be used to obtain
#' a subset of data for further analysis or modeling.
#'
#' Two basic specializations of data_sample are sample_random and sample_stratified.
#' They provide random sampling and stratified sampling, respectively.
#'
#' Data sample provides both training and testing partitioning (train_test) and
#' k-fold partitioning (k_fold) of data.
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
#'@description Partitions a dataset into training and test sets using a specified sampling method
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
#'@description k-fold partition of a dataset using a sampling method
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


