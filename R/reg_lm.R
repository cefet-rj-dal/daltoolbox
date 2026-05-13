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
#'  eval_simple <- evaluate(model_simple, Boston$medv, pred_simple)
#'  eval_simple$metrics
#'
#'  # Polynomial regression (degree 2)
#'  model_poly <- reg_lm(formula = medv ~ poly(lstat, 2, raw = TRUE))
#'  model_poly <- fit(model_poly, Boston)
#'  pred_poly <- predict(model_poly, Boston)
#'  eval_poly <- evaluate(model_poly, Boston$medv, pred_poly)
#'  eval_poly$metrics
#'
#'  # Multiple regression
#'  model_multi <- reg_lm(formula = medv ~ lstat + rm + ptratio)
#'  model_multi <- fit(model_multi, Boston)
#'  pred_multi <- predict(model_multi, Boston)
#'  eval_multi <- evaluate(model_multi, Boston$medv, pred_multi)
#'  eval_multi$metrics
#'}
#'@export
reg_lm <- function(formula = NULL, attribute = NULL, features = NULL) {
  obj <- regression(attribute)
  obj$formula <- formula
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
    obj <- fit.predictor(obj, data)
  } else {
    formula_vars <- all.vars(obj$formula)
    obj$attribute <- formula_vars[1]
    obj$x <- intersect(formula_vars[-1], names(data))
    if (length(obj$x) == 0) {
      obj$x <- setdiff(names(data), obj$attribute)
    }
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
