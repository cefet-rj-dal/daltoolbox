#'@title Linear regression (lm)
#'@description Linear regression using `stats::lm`.
#'@param formula optional regression formula (e.g., y ~ x1 + x2).
#'@param attribute target attribute name (used when formula is NULL)
#'@param features optional vector of feature names (used when formula is NULL)
#'@return returns a `reg_lm` object
#'@examples
#'if (requireNamespace("MASS", quietly = TRUE)) {
#'  data(Boston, package = "MASS")
#'
#'  # Simple linear regression
#'  model_simple <- reg_lm(formula = medv ~ lstat)
#'  model_simple <- fit(model_simple, Boston)
#'  pred_simple <- predict(model_simple, Boston)
#'  head(pred_simple)
#'
#'  # Polynomial regression (degree 2)
#'  model_poly <- reg_lm(formula = medv ~ poly(lstat, 2, raw = TRUE))
#'  model_poly <- fit(model_poly, Boston)
#'  pred_poly <- predict(model_poly, Boston)
#'  head(pred_poly)
#'
#'  # Multiple regression
#'  model_multi <- reg_lm(formula = medv ~ lstat + rm + ptratio)
#'  model_multi <- fit(model_multi, Boston)
#'  pred_multi <- predict(model_multi, Boston)
#'  head(pred_multi)
#'}
#'@export
reg_lm <- function(formula = NULL, attribute = NULL, features = NULL) {
  obj <- dal_learner()
  obj$formula <- formula
  obj$attribute <- attribute
  obj$features <- features
  obj$model <- NULL
  class(obj) <- append("reg_lm", class(obj))
  return(obj)
}

#'@importFrom stats lm
#'@exportS3Method fit reg_lm
fit.reg_lm <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  if (is.null(obj$formula)) {
    if (is.null(obj$attribute)) {
      stop("reg_lm: provide 'formula' or 'attribute'.")
    }
    features <- obj$features
    if (is.null(features)) {
      features <- setdiff(names(data), obj$attribute)
    }
    obj$formula <- stats::formula(
      paste(obj$attribute, "~", paste(features, collapse = " + "))
    )
  }
  obj$model <- stats::lm(obj$formula, data = data)
  return(obj)
}

#'@importFrom stats predict
#'@exportS3Method predict reg_lm
predict.reg_lm <- function(object, newdata, ...) {
  newdata <- adjust_data.frame(newdata)
  stats::predict(object$model, newdata = newdata)
}
