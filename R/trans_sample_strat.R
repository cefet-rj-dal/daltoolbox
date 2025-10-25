#'@title  Stratified Random Sampling
#'@description The sample_stratified function in R is used to generate a stratified random sample from a given dataset.
#'Stratified sampling is a statistical method that is used when the population is divided into non-overlapping subgroups or strata, and a sample is selected from each stratum to represent the entire population.
#'In stratified sampling, the sample is selected in such a way that it is representative of the entire population and the variability within each stratum is minimized.
#'@param attribute attribute target to model building
#'@return returns an object of class `sample_stratified`
#'@examples
#'#using stratified sampling
#'sample <- sample_stratified("Species")
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
sample_stratified <- function(attribute) {
  obj <- sample_random()
  obj$attribute <- attribute
  class(obj) <- append("sample_stratified", class(obj))
  return(obj)
}

#'@importFrom caret createDataPartition
#'@exportS3Method train_test sample_stratified
train_test.sample_stratified <- function(obj, data, perc=0.8, ...) {
  predictors_name <- setdiff(colnames(data), obj$attribute)
  predictand <- data[,obj$attribute]

  # maintain class distribution in train/test via stratification
  idx <- caret::createDataPartition(predictand, p=perc, list=FALSE)
  train <- data[idx,]
  test <- data[-idx,]
  return (list(train=train, test=test))
}

#'@exportS3Method k_fold sample_stratified
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
  return (folds)
}
