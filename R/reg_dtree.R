#'@title Decision Tree for regression
#'@description Creates a regression object that
#' uses the Decision Tree method for regression
#' It wraps the tree library.
#'@param attribute attribute target to model building.
#'@return returns a decision tree regression object
#'@examples
#'data(Boston)
#'model <- reg_dtree("medv")
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
reg_dtree <- function(attribute) {
  obj <- regression(attribute)

  class(obj) <- append("reg_dtree", class(obj))
  return(obj)
}

#'@importFrom tree tree
#'@exportS3Method fit reg_dtree
fit.reg_dtree <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  obj <- fit.predictor(obj, data)

  regression <- formula(paste(obj$attribute, "  ~ ."))
  obj$model <- tree::tree(regression, data)

  return(obj)
}

#'@exportS3Method predict reg_dtree
predict.reg_dtree <- function(object, x, ...) {
  x <- adjust_data.frame(x)
  x <- x[,object$x]
  prediction <- predict(object$model, x, type="vector")
  return(prediction)
}
