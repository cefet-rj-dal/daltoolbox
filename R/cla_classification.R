#'@title Classification base class
#'@description Ancestor class for classification models providing common fields (target attribute and levels)
#' and evaluation helpers.
#'@param attribute attribute target to model building
#'@param slevels possible values for the target classification
#'@return returns a classification object
#'@examples
#'#See ?cla_dtree for a classification example using a decision tree
#'@export
classification <- function(attribute, slevels = NULL) {
  obj <- predictor()
  class(obj) <- append("classification", class(obj))
  obj$attribute <- attribute
  obj$slevels <- slevels
  # internal numeric indices for class levels
  obj$ilevels <- if (is.null(slevels)) integer(0) else seq_along(slevels)
  return(obj)
}

prepare_classification_data <- function(obj, data) {
  data <- adjust_data.frame(data)
  attr <- obj$attribute
  if (is.null(attr) || !attr %in% names(data)) {
    stop(sprintf("%s: attribute not found in data.", class(obj)[1]), call. = FALSE)
  }

  if (is.null(obj$slevels)) {
    y <- data[[attr]]
    if (is.factor(y)) {
      obj$slevels <- levels(y)
    } else {
      obj$slevels <- unique(as.character(y))
    }
  }

  obj$ilevels <- seq_along(obj$slevels)
  data[, attr] <- adjust_factor(data[, attr], obj$ilevels, obj$slevels)
  obj <- fit.predictor(obj, data)

  list(obj = obj, data = data)
}

prediction_as_factor <- function(prediction, slevels) {
  if (is.factor(prediction)) {
    return(factor(prediction, levels = slevels))
  }

  if (is.data.frame(prediction) || is.matrix(prediction)) {
    prediction <- as.matrix(prediction)
    if (!is.null(colnames(prediction))) {
      prediction <- prediction[, slevels, drop = FALSE]
    }
    y <- apply(prediction, 1, nnet::which.is.max)
    return(factor(slevels[y], slevels))
  }

  factor(prediction, levels = slevels)
}

prediction_as_probability <- function(prediction, slevels) {
  if (is.factor(prediction) || is.vector(prediction)) {
    return(adjust_class_label(factor(prediction, levels = slevels)))
  }

  if (is.data.frame(prediction) || is.matrix(prediction)) {
    prediction <- as.matrix(prediction)
    if (!is.null(colnames(prediction))) {
      prediction <- prediction[, slevels, drop = FALSE]
    }
    return(prediction)
  }

  adjust_class_label(factor(prediction, levels = slevels))
}


#'@importFrom nnet which.is.max
#'@exportS3Method evaluate classification
evaluate.classification <- function(obj, data, prediction, ref = 1, ...) {
  # accept either labels or probability matrix for data and prediction
  data_f <- prediction_as_factor(data, obj$slevels)
  pred_f <- prediction_as_factor(prediction, obj$slevels)
  pred_scores <- prediction_as_probability(prediction, obj$slevels)
  pred_p <- apply(pred_scores, 1, max)

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

