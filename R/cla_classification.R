#'@title classification
#'@description Ancestor class for classification problems using MLmetrics nnet
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
  obj$ilevels <- 1:length(slevels)
  return(obj)
}


#'@import MLmetrics nnet
#'@export
evaluate.classification <- function(obj, data, prediction, ...) {
  result <- list(data=data, prediction=prediction)

  adjust_predictions <- function(predictions) {
    predictions_i <- matrix(rep.int(0, nrow(predictions)*ncol(predictions)), nrow=nrow(predictions), ncol=ncol(predictions))
    y <- apply(predictions, 1, nnet::which.is.max)
    for(i in unique(y)) {
      predictions_i[y==i,i] <- 1
    }
    return(predictions_i)
  }
  predictions <- adjust_predictions(result$prediction)
  result$conf_mat <- MLmetrics::ConfusionMatrix(data, predictions)
  result$accuracy <- MLmetrics::Accuracy(y_pred = predictions, y_true = data)
  result$f1 <- MLmetrics::F1_Score(y_pred = predictions, y_true = data, positive = 1)
  result$sensitivity <- MLmetrics::Sensitivity(y_pred = predictions, y_true = data, positive = 1)
  result$specificity <- MLmetrics::Specificity(y_pred = predictions, y_true = data, positive = 1)
  result$precision <- MLmetrics::Precision(y_pred = predictions, y_true = data, positive = 1)
  result$recall <- MLmetrics::Recall(y_pred = predictions, y_true = data, positive = 1)
  result$metrics <- data.frame(accuracy=result$accuracy, f1=result$f1,
    sensitivity=result$sensitivity, specificity=result$specificity,
    precision=result$precision, recall=result$recall)

  return(result)
}

