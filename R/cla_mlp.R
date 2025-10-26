#'@title MLP for classification
#'@description Multi-Layer Perceptron classifier using `nnet::nnet` (single hidden layer).
#'@details Uses softmax output with one‑hot targets from `adjust_class_label`. `size` controls hidden units and
#' `decay` applies L2 regularization. Features should be scaled.
#'@param attribute attribute target to model building
#'@param slevels possible values for the target classification
#'@param size number of nodes that will be used in the hidden layer
#'@param decay how quickly it decreases in gradient descent
#'@param maxit maximum iterations
#'@return returns a classification object
#'@references
#' Rumelhart, D., Hinton, G., Williams, R. (1986). Learning representations by back‑propagating errors.
#' Bishop, C. M. (1995). Neural Networks for Pattern Recognition.
#'@examples
#'data(iris)
#'slevels <- levels(iris$Species)
#'model <- cla_mlp("Species", slevels, size=3, decay=0.03)
#'
#'# preparing dataset for random sampling
#'sr <- sample_random()
#'sr <- train_test(sr, iris)
#'train <- sr$train
#'test <- sr$test
#'
#'model <- fit(model, train)
#'
#'prediction <- predict(model, test)
#'predictand <- adjust_class_label(test[,"Species"])
#'test_eval <- evaluate(model, predictand, prediction)
#'test_eval$metrics
#'@export
cla_mlp <- function(attribute, slevels, size=NULL, decay=0.1, maxit=1000) {
  obj <- classification(attribute, slevels)
  obj$maxit <- maxit
  obj$size <- size
  obj$decay <- decay

  class(obj) <- append("cla_mlp", class(obj))
  return(obj)
}

#'@importFrom nnet nnet
#'@exportS3Method fit cla_mlp
fit.cla_mlp <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  # turn target into factor with desired label order
  data[,obj$attribute] <- adjust_factor(data[,obj$attribute], obj$ilevels, obj$slevels)
  # record predictors for consistent inference
  obj <- fit.predictor(obj, data)

  # sensible default for hidden size when not provided
  if (is.null(obj$size))
    obj$size <- ceiling(sqrt(ncol(data)))

  x <- data[,obj$x, drop = FALSE]
  y <- data[,obj$attribute]

  # train MLP with one-hot targets
  obj$model <- nnet::nnet(x = x, y = adjust_class_label(y), size=obj$size, decay=obj$decay, maxit=obj$maxit, trace=FALSE)

  return(obj)
}

#'@exportS3Method predict cla_mlp
predict.cla_mlp  <- function(object, x, ...) {
  x <- adjust_data.frame(x)
  # ensure same predictors as used for training
  x <- x[,object$x, drop = FALSE]

  prediction <- predict(object$model, x, type="raw")
  prediction <- as.data.frame(prediction)
  colnames(prediction) <- object$slevels

  return(prediction)
}
