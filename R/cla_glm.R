#'@title Logistic regression (GLM)
#'@description Logistic regression classifier using `stats::glm` with binomial family.
#'@param attribute target attribute name
#'@param positive positive class label
#'@param features optional vector of feature names (default: all except attribute)
#'@param threshold probability threshold for positive class
#'@return returns a `cla_glm` object
#'@examples
#'data(iris)
#'iris_bin <- iris
#'iris_bin$IsVersicolor <- factor(ifelse(
#'  iris_bin$Species == "versicolor",
#'  "versicolor",
#'  "not_versicolor"
#'))
#'model <- cla_glm("IsVersicolor", positive = "versicolor")
#'model <- suppressWarnings(fit(model, iris_bin))
#'pred <- predict(model, iris_bin)
#'eval <- evaluate(model, adjust_class_label(iris_bin$IsVersicolor), pred)
#'eval$metrics
#'@references
#' Nelder, J. A. and Wedderburn, R. W. M. (1972). Generalized Linear Models. Journal of the Royal Statistical Society, Series A, 135(3):370-384.
#' Hosmer, D. W. and Lemeshow, S. (2000). Applied Logistic Regression (2nd ed.). Wiley.
#'@export
cla_glm <- function(attribute, positive, features = NULL, threshold = 0.5) {
  obj <- classification(attribute)
  obj$positive <- positive
  obj$features <- features
  obj$threshold <- threshold
  obj$model <- NULL
  class(obj) <- append("cla_glm", class(obj))
  return(obj)
}

#'@importFrom stats glm binomial
#'@exportS3Method fit cla_glm
fit.cla_glm <- function(obj, data, ...) {
  prepared <- prepare_classification_data(obj, data)
  obj <- prepared$obj
  data <- prepared$data
  attr <- obj$attribute
  features <- obj$features
  if (is.null(features)) {
    features <- obj$x
  }
  if (length(obj$slevels) != 2) {
    stop("cla_glm supports only binary classification.", call. = FALSE)
  }
  if (!obj$positive %in% obj$slevels) {
    stop("cla_glm: positive class must be one of the target levels.", call. = FALSE)
  }
  # binomial glm models P(Y = last factor level); relevel so the positive class is
  # that last level. Otherwise predict() would assign P(other class) to the positive
  # column whenever `positive` is not the last level, inverting the prediction.
  neg <- setdiff(obj$slevels, obj$positive)
  data[[attr]] <- stats::relevel(
    factor(data[[attr]], levels = obj$slevels), ref = neg
  )
  formula <- stats::formula(
    paste(attr, "~", paste(features, collapse = " + "))
  )
  obj$model <- stats::glm(formula, data = data, family = binomial)
  return(obj)
}

#'@importFrom stats predict
#'@exportS3Method predict cla_glm
predict.cla_glm <- function(object, newdata, ...) {
  newdata <- adjust_data.frame(newdata)
  x <- newdata[, object$x, drop = FALSE]
  prob <- stats::predict(object$model, newdata = x, type = "response")
  prediction <- matrix(0, nrow = length(prob), ncol = length(object$slevels))
  colnames(prediction) <- object$slevels
  pos_idx <- match(object$positive, object$slevels)
  neg_idx <- setdiff(seq_along(object$slevels), pos_idx)
  prediction[, pos_idx] <- prob
  prediction[, neg_idx] <- 1 - prob
  as.data.frame(prediction)
}
