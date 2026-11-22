#' @title Stratified Sampling
#' @description Train/test split and k-fold partitioning that preserve the
#'   target class proportions.
#' @details Use this sampler when the response distribution matters, especially
#'   in imbalanced classification problems. Compared with simple random
#'   sampling, it reduces the chance that one split will overrepresent or
#'   underrepresent a class by accident.
#' @param attribute Name of the target attribute whose class proportions should
#'   be preserved.
#' @return An object of class `sample_stratified`.
#' @examples
#' # using stratified sampling
#' sample <- sample_stratified("Species")
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
sample_stratified <- function(attribute) {
  obj <- sample_random()
  obj$attribute <- attribute
  class(obj) <- append("sample_stratified", class(obj))
  return(obj)
}

# Stratified partition: per class, draw round(perc * n_class) indices for the train set,
# preserving the class proportions.
strat_partition <- function(y, perc) {
  y <- as.character(y)
  idx <- integer(0)
  for (cls in unique(y)) {
    ci <- which(y == cls)
    n <- length(ci)
    n_train <- max(1L, min(as.integer(round(perc * n)), n))
    idx <- c(idx, if (n_train >= n) ci else sample(ci, size = n_train))
  }
  sort(idx)
}

#' @exportS3Method train_test sample_stratified
train_test.sample_stratified <- function(obj, data, perc = 0.8, ...) {
  predictand <- data[, obj$attribute]

  # maintain class distribution in train/test via stratification
  idx <- strat_partition(predictand, perc)
  train <- data[idx, ]
  test <- data[-idx, ]
  return(list(train = train, test = test))
}

#' @exportS3Method k_fold sample_stratified
k_fold.sample_stratified <- function(obj, data, k) {
  folds <- list()
  samp <- list()
  p <- 1.0 / k
  while (k > 1) {
    # iteratively split off 1/k of remaining data preserving strata
    samp <- train_test.sample_stratified(obj, data, p)
    data <- samp$test
    folds <- append(folds, list(samp$train))
    k = k - 1
    p = 1.0 / k
  }
  folds <- append(folds, list(samp$test))
  return(folds)
}
