#'@title Time Series Sample
#'@description Separates the `ts_data` into training and test.
#'It separates the test size from the last observations minus an offset.
#'The offset is important to allow replication under different recent origins.
#'The data for train uses the number of rows of a `ts_data` minus the test size and offset.
#'@param ts time series.
#'@param test_size integer: size of test data (default = 1).
#'@param offset integer: starting point (default = 0).
#'@return returns a list with the two samples
#'@examples
#'#setting up a ts_data
#'data(sin_data)
#'ts <- ts_data(sin_data$y, 10)
#'
#'#separating into train and test
#'test_size <- 3
#'samp <- ts_sample(ts, test_size)
#'
#'#first five rows from training data
#'ts_head(samp$train, 5)
#'
#'#last five rows from training data
#'ts_head(samp$train[-c(1:(nrow(samp$train)-5)),])
#'
#'#testing data
#'ts_head(samp$test)
#'@export
ts_sample <- function(ts, test_size=1, offset=0) {
  offset <- nrow(ts) - test_size - offset
  train <- ts[1:offset, ]
  test <- ts[(offset+1):(offset+test_size),]
  colnames(test) <- colnames(train)
  samp <- list(train = train, test = test)
  attr(samp, "class") <- "ts_sample"
  return(samp)
}


