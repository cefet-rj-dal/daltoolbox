#'@title Feature selection by stepwise model selection
#'@description Select features using stepwise search over generalized linear models.
#'@details Supports forward, backward, and both directions via `stats::step`.
#'@param attribute target attribute name
#'@param features optional vector of feature names (default: all columns except `attribute`)
#'@param direction stepwise direction: "forward", "backward", or "both"
#'@param family glm family passed to `stats::glm` (default: `binomial`)
#'@param trace level of tracing from `stats::step`
#'@return returns an object of class `feature_selection_stepwise`
#'@examples
#'data(iris)
#'fg <- feature_generation(
#'  IsVersicolor = ifelse(Species == "versicolor", "versicolor", "not_versicolor")
#')
#'iris_bin <- transform(fg, iris)
#'iris_bin$IsVersicolor <- factor(iris_bin$IsVersicolor)
#'fs <- feature_selection_stepwise("IsVersicolor", direction = "forward")
#'fs <- fit(fs, iris_bin)
#'fs$selected
#'transform(fs, iris_bin) |> names()
#'@export
feature_selection_stepwise <- function(attribute, features = NULL, direction = "forward", family = stats::binomial, trace = 0) {
  obj <- dal_transform()
  obj$attribute <- attribute
  obj$features <- features
  obj$direction <- direction
  obj$family <- family
  obj$trace <- trace
  class(obj) <- append("feature_selection_stepwise", class(obj))
  return(obj)
}

#'@importFrom stats glm step formula terms binomial
#'@exportS3Method fit feature_selection_stepwise
fit.feature_selection_stepwise <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  attr <- obj$attribute
  if (is.null(attr) || !attr %in% names(data)) {
    stop("feature_selection_stepwise: attribute not found in data.")
  }

  features <- obj$features
  if (is.null(features)) {
    features <- setdiff(names(data), attr)
  }
  features <- intersect(features, names(data))
  obj$features <- features

  if (length(features) == 0) {
    obj$ranking <- data.frame(feature = character(0), score = numeric(0), stringsAsFactors = FALSE)
    obj$selected <- character(0)
    return(obj)
  }

  full_formula <- stats::formula(
    paste(attr, "~", paste(features, collapse = " + "))
  )
  null_formula <- stats::formula(
    paste(attr, "~ 1")
  )

  direction <- match.arg(obj$direction, c("forward", "backward", "both"))
  if (direction == "backward") {
    base_model <- stats::glm(full_formula, data = data, family = obj$family)
    step_model <- stats::step(base_model, direction = direction, trace = obj$trace)
  } else {
    null_model <- stats::glm(null_formula, data = data, family = obj$family)
    full_model <- stats::glm(full_formula, data = data, family = obj$family)
    step_model <- stats::step(
      null_model,
      scope = list(lower = null_model, upper = full_model),
      direction = direction,
      trace = obj$trace
    )
  }

  selected <- attr(stats::terms(step_model), "term.labels")
  ranking <- data.frame(
    feature = selected,
    score = seq_along(selected),
    stringsAsFactors = FALSE
  )

  obj$model <- step_model
  obj$selected <- selected
  obj$ranking <- ranking
  return(obj)
}

#'@exportS3Method transform feature_selection_stepwise
transform.feature_selection_stepwise <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  if (is.null(obj$selected)) {
    stop("feature_selection_stepwise: call fit() before transform().")
  }
  keep <- c(obj$attribute, obj$selected)
  keep <- intersect(keep, names(data))
  data <- data[, keep, drop = FALSE]
  return(data)
}
