#'@title Sample Random
#'@description The sample_random function in R is used to
#' generate a random sample of specified size from a given data set.
#'@return returns an object of class `sample_random
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
sample_random <- function() {
  obj <- data_sample()
  class(obj) <- append("sample_random", class(obj))
  return(obj)
}

#'@exportS3Method train_test sample_random
train_test.sample_random <- function(obj, data, perc=0.8, ...) {
  # randomly sample row indices for the training set
  idx <- base::sample(1:nrow(data),as.integer(perc*nrow(data)))
  train <- data[idx,]
  test <- data[-idx,]
  return (list(train=train, test=test))
}

#'@exportS3Method k_fold sample_random
k_fold.sample_random <- function(obj, data, k) {
  folds <- list()
  samp <- list()
  p <- 1.0 / k
  while (k > 1) {
    # iteratively split off 1/k of remaining data as a fold
    samp <- train_test.sample_random(obj, data, p)
    data <- samp$test
    folds <- append(folds, list(samp$train))
    k = k - 1
    p = 1.0 / k
  }
  folds <- append(folds, list(samp$test))
  return (folds)
}

#'@title k-fold training and test partition object
#'@description Splits a dataset into training and test sets based on k-fold cross-validation.
#'The function takes a list of data partitions (folds) and a specified fold index k.
#'It returns the data corresponding to the k-th fold as the test set, and combines all other folds to form the training set.
#'@param folds data partitioned into folds
#'@param k k-fold for test set, all reminder for training set
#'@return returns a list with two elements:
#' \itemize{
#'   \item train: A data frame containing the combined data from all folds except the k-th fold, used as the training set.
#'   \item test: A data frame corresponding to the k-th fold, used as the test set.
#' }
#'@examples
#'# Create k-fold partitions of a dataset (e.g., iris)
#'folds <- k_fold(sample_random(), iris, k = 5)
#'
#'# Use the first fold as the test set and combine the remaining folds for the training set
#'train_test_split <- train_test_from_folds(folds, k = 1)
#'
#'# Display the training set
#'head(train_test_split$train)
#'
#'# Display the test set
#'head(train_test_split$test)
#'@export
train_test_from_folds <- function(folds, k) {
  test <- folds[[k]]
  train <- NULL
  # concatenate all folds except k into the training set
  for (i in 1:length(folds)) {
    if (i != k)
      train <- rbind(train, folds[[i]])
  }
  return (list(train=train, test=test))
}
