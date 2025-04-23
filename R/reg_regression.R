#'@title Regression
#'@description Ancestor class for regression problems.
#'This ancestor class is used to define and manage the target attribute for regression tasks.
#'@param attribute attribute target to model building
#'@return returns a regression object
#'@examples
#'#See ?reg_dtree for a regression example using a decision tree
#'@export
regression <- function(attribute) {
  obj <- predictor()
  obj$attribute <- attribute
  class(obj) <- append("regression", class(obj))
  return(obj)
}

#'@exportS3Method evaluate regression
evaluate.regression <- function(obj, values, prediction, ...) {
  MSE <- function (actual, prediction) {
    if (length(actual) != length(prediction))
      stop("actual and prediction have different lengths")
    n <- length(actual)
    res <- mean((actual - prediction)^2)
    res
  }

  sMAPE <- function (actual, prediction) {
    if (length(actual) != length(prediction))
      stop("actual and prediction have different lengths")
    n <- length(actual)
    res <- (1/n) * sum(abs(actual - prediction)/((abs(actual) +
                                                    abs(prediction))/2))
    res
  }

  R2 <- function (actual, prediction) {
    if (length(actual) != length(prediction))
      stop("actual and prediction have different lengths")
    res <-  1 - sum((prediction - actual)^2)/sum((mean(actual) - actual)^2)
    res
  }


  result <- list(values=values, prediction=prediction)

  result$smape <- sMAPE(values, prediction)
  result$mse <- MSE(values, prediction)
  result$R2 <- R2(values, prediction)

  result$metrics <- data.frame(mse=result$mse, smape=result$smape, R2 = result$R2)

  return(result)
}
