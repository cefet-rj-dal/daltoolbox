#'@title Hierarchy mapping by cut
#'@description Create a categorical hierarchy from a numeric attribute using cut points.
#'@param attribute numeric attribute to discretize
#'@param breaks numeric breakpoints for `cut`
#'@param labels optional labels for the cut intervals
#'@param new_attribute name of the new attribute (default: "attribute.Level")
#'@return returns an object of class `hierarchy_cut`
#'@examples
#'data(iris)
#'hc <- hierarchy_cut(
#'  "Sepal.Length",
#'  breaks = c(-Inf, 5.5, 6.5, Inf),
#'  labels = c("baixo", "medio", "alto")
#')
#'iris_h <- transform(hc, iris)
#'table(iris_h$Sepal.Length.Level)
#'@export
hierarchy_cut <- function(attribute, breaks, labels = NULL, new_attribute = NULL) {
  obj <- dal_transform()
  obj$attribute <- attribute
  obj$breaks <- breaks
  obj$labels <- labels
  obj$new_attribute <- new_attribute
  class(obj) <- append("hierarchy_cut", class(obj))
  return(obj)
}

#'@exportS3Method transform hierarchy_cut
transform.hierarchy_cut <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  attribute <- obj$attribute
  if (is.null(attribute) || !attribute %in% names(data)) {
    stop("hierarchy_cut: 'attribute' must be a valid column name in data.")
  }
  new_attribute <- obj$new_attribute
  if (is.null(new_attribute)) {
    new_attribute <- paste0(attribute, ".Level")
  }

  data[[new_attribute]] <- cut(
    data[[attribute]],
    breaks = obj$breaks,
    labels = obj$labels,
    include.lowest = TRUE
  )
  return(data)
}
