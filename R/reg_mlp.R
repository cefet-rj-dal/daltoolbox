#'@title MLP for regression
#'@description Creates a regression object that
#' uses the Multi-Layer Perceptron (MLP) method.
#' It wraps the nnet library.
#'@param attribute attribute target to model building
#'@param size number of neurons in hidden layers
#'@param decay decay learning rate
#'@param maxit number of maximum iterations for training
#'@return returns a object of class `reg_mlp`
#'@examples
#'data(Boston)
#'model <- reg_mlp("medv", size=5, decay=0.54)
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
reg_mlp <- function(attribute, size=NULL, decay=0.05, maxit=1000) {
  obj <- regression(attribute)
  obj$maxit <- maxit
  obj$size <- size
  obj$decay <- decay
  class(obj) <- append("reg_mlp", class(obj))
  return(obj)
}

#'@importFrom nnet nnet
#'@export
fit.reg_mlp <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  obj <- fit.predictor(obj, data)

  if (is.null(obj$size))
    obj$size <- ceiling(ncol(data)/3)

  x <- data[,obj$x]
  y <- data[,obj$attribute]

  obj$model <- nnet::nnet(x = x, y = y, size = obj$size, decay = obj$decay, maxit=obj$maxit, linout=TRUE, trace = FALSE)

  return(obj)
}

#'@export
predict.reg_mlp  <- function(object, x, ...) {
  x <- adjust_data.frame(x)
  x <- x[,object$x]
  prediction <- predict(object$model, x)
  return(prediction)
}
