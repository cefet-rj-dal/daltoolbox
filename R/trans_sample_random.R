#' @title Random Sampling
#' @description Train/test split and k-fold partitioning by simple random
#'   sampling.
#' @details This is the simplest partitioning strategy in `daltoolbox`. It is a
#'   good starting point when the dataset is reasonably balanced and the user
#'   wants to understand the train/test workflow before moving to stratified or
#'   more specialized sampling schemes.
#' @return An object of class `sample_random`.
#' @examples
#' # using random sampling
#' sample <- sample_random()
#' tt <- train_test(sample, iris)
#'
#' # distribution of train
#' table(tt$train$Species)
#'
#' # preparing dataset into four folds
#' folds <- k_fold(sample, iris, 4)
#'
#' # distribution of folds
#' tbl <- NULL
#' for (f in folds) {
#'   tbl <- rbind(tbl, table(f$Species))
#' }
#' head(tbl)
#' @export
sample_random <- function() {
  obj <- data_sample()
  class(obj) <- append("sample_random", class(obj))
  return(obj)
}

#' @exportS3Method train_test sample_random
train_test.sample_random <- function(obj, data, perc = 0.8, ...) {
  # randomly sample row indices for the training set
  idx <- base::sample(1:nrow(data), as.integer(perc * nrow(data)))
  train <- data[idx,]
  test <- data[-idx,]
  return(list(train = train, test = test))
}

#' @exportS3Method k_fold sample_random
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
  return(folds)
}

#' @title k-Fold Training and Test Partition Object
#' @description Build one train/test split from a list of folds produced by
#'   cross-validation.
#' @details The k-th fold is returned as the test set and all remaining folds
#'   are concatenated into the training set. This helper makes manual
#'   cross-validation loops easier to explain and reproduce.
#' @param folds Data already partitioned into folds.
#' @param k Fold index to be used as the test set.
#' @return A list with two elements:
#' \itemize{
#'   \item train: data frame containing all folds except the k-th.
#'   \item test: data frame corresponding to the k-th fold.
#' }
#' @examples
#' # Create k-fold partitions of a dataset (e.g., iris)
#' folds <- k_fold(sample_random(), iris, k = 5)
#'
#' # Use the first fold as the test set and combine the remaining folds
#' # for the training set
#' train_test_split <- train_test_from_folds(folds, k = 1)
#'
#' # Display the training set
#' head(train_test_split$train)
#'
#' # Display the test set
#' head(train_test_split$test)
#' @export
train_test_from_folds <- function(folds, k) {
  test <- folds[[k]]
  train <- NULL
  # concatenate all folds except k into the training set
  for (i in 1:length(folds)) {
    if (i != k)
      train <- rbind(train, folds[[i]])
  }
  return(list(train = train, test = test))
}
