#'@title Random Forest for regression
#'@description Regression via Random Forests, an ensemble of decision trees trained
#' on bootstrap samples with random feature subsetting at each split. This wrapper
#' uses the `randomForest` package API.
#'@details Random Forests reduce variance and are robust to overfitting on tabular data.
#' Key hyperparameters are the number of trees (`ntree`), the number of variables tried at
#' each split (`mtry`), and the minimum node size (`nodesize`).
#'@param attribute attribute target to model building
#'@param nodesize node size
#'@param ntree number of trees
#'@param mtry number of attributes to build tree
#'@return returns an object of class `reg_rf`obj
#'@references
#' Breiman, L. (2001). Random Forests. Machine Learning 45(1):5–32.
#' Liaw, A. and Wiener, M. (2002). Classification and Regression by randomForest. R News.
#'@examples
#'data(Boston)
#'model <- reg_rf("medv", ntree=10)
#'
#'# preparing dataset for random sampling
#'sr <- sample_random()
#'sr <- train_test(sr, Boston)
#'train <- sr$train
#'test <- sr$test
#'
#'model <- fit(model, train)
#'
#'test_prediction <- predict(model, test)
#'test_predictand <- test[,"medv"]
#'test_eval <- evaluate(model, test_predictand, test_prediction)
#'test_eval$metrics
#'@export
reg_rf <- function(attribute, nodesize = 1, ntree = 10, mtry = NULL) {
  obj <- regression(attribute)

  obj$nodesize <- nodesize
  obj$ntree <- ntree
  obj$mtry <- mtry

  class(obj) <- append("reg_rf", class(obj))
  return(obj)
}

#'@importFrom randomForest randomForest
#'@exportS3Method fit reg_rf
fit.reg_rf <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  # record feature columns (exclude target attribute)
  obj <- fit.predictor(obj, data)

  # default mtry heuristic if not provided
  if (is.null(obj$mtry))
    obj$mtry <- ceiling(ncol(data)/3)

  # split into features (x) and target (y)
  x <- data[,obj$x]
  y <- data[,obj$attribute]

  obj$model <- randomForest::randomForest(x = x, y = y, nodesize = obj$nodesize, mtry=obj$mtry, ntree=obj$ntree)

  return(obj)
}

#'@exportS3Method predict reg_rf
predict.reg_rf  <- function(object, x, ...) {
  x <- adjust_data.frame(x)
  # ensure prediction uses the same feature columns seen in training
  x <- x[,object$x]
  prediction <- predict(object$model, x)
  return(prediction)
}
