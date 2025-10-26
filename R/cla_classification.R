#'@title Classification base class
#'@description Ancestor class for classification models providing common fields (target attribute and levels)
#' and evaluation helpers.
#'@param attribute attribute target to model building
#'@param slevels possible values for the target classification
#'@return returns a classification object
#'@examples
#'#See ?cla_dtree for a classification example using a decision tree
#'@export
classification <- function(attribute, slevels) {
  obj <- predictor()
  class(obj) <- append("classification", class(obj))
  obj$attribute <- attribute
  obj$slevels <- slevels
  # internal numeric indices for class levels
  obj$ilevels <- 1:length(slevels)
  return(obj)
}


#'@importFrom nnet which.is.max
#'@exportS3Method evaluate classification
evaluate.classification <- function(obj, data, prediction, ref = 1, ...) {
  variables_as_factor <- function(prediction, s_levels) {
    y <- apply(prediction, 1, nnet::which.is.max)
    yfact <- factor(s_levels[y], s_levels)
    return(yfact)
  }
  variables_as_probability <- function(prediction) {
    y <- apply(prediction, 1, max)
    return(y)
  }


  # accept either factor labels or probability matrix for data
  data_f <- data
  if (!is.factor(data_f))
    data_f <- variables_as_factor(data_f, obj$slevels)

  pred_f <- variables_as_factor(prediction, obj$slevels)
  pred_p <- variables_as_probability(prediction)

  result <- list(data=data_f, prediction=pred_f, probability=pred_p)

  metrics <- list()

  metrics$accuracy <- sum(data_f == pred_f)/length(data_f)

  data <- data_f == obj$slevels[ref]
  pred <- pred_f == obj$slevels[ref]

  metrics$TP <- sum(data == 1 & pred == 1)
  metrics$TN <- sum(data == 0 & pred == 0)
  metrics$FP <- sum(data == 0 & pred == 1)
  metrics$FN <- sum(data == 1 & pred == 0)

  metrics$precision <- metrics$TP/(metrics$TP+metrics$FP)
  metrics$recall <- metrics$TP/(metrics$TP+metrics$FN)

  metrics$sensitivity <- metrics$recall
  metrics$specificity <- metrics$TN/(metrics$TN+metrics$FP)

  metrics$f1 <- 2*(metrics$precision*metrics$recall)/(metrics$precision+metrics$recall)

  # return metrics as a single-row data.frame
  result$metrics <- as.data.frame(metrics)

  return(result)
}

