#'@title Feature generation
#'@description Create new features from existing columns using named expressions.
#'@param ... named expressions that compute new features
#'@return returns an object of class `feature_generation`
#'@examples
#'data(iris)
#'gen <- feature_generation(
#'  Sepal.Area = Sepal.Length * Sepal.Width,
#'  Petal.Area = Petal.Length * Petal.Width,
#'  Sepal.Ratio = Sepal.Length / Sepal.Width
#')
#'iris_feat <- transform(gen, iris)
#'head(iris_feat)
#'@export
feature_generation <- function(...) {
  obj <- dal_transform()
  obj$exprs <- as.list(substitute(list(...)))[-1]
  class(obj) <- append("feature_generation", class(obj))
  return(obj)
}

#'@exportS3Method transform feature_generation
transform.feature_generation <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  exprs <- obj$exprs
  if (length(exprs) == 0) {
    return(data)
  }
  if (is.null(names(exprs)) || any(names(exprs) == "")) {
    stop("feature_generation: all generated features must be named.")
  }
  for (nm in names(exprs)) {
    data[[nm]] <- eval(exprs[[nm]], data, parent.frame())
  }
  return(data)
}
