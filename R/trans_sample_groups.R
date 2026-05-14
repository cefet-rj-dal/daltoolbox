#'@title Group sampling
#'@description Sample entire groups defined by a categorical attribute.
#' In sampling theory, this design is known as cluster sampling
#' (also called one-stage cluster sampling or sampling by groups).
#' The groups are assumed to be pre-defined in the data; this function
#' does not infer groups with clustering algorithms such as k-means.
#'@param attribute group-defining attribute name
#'@param n_groups number of groups to sample
#'@return returns an object of class `sample_groups`
#'@examples
#'data(iris)
#'sc <- sample_groups("Species", n_groups = 2)
#'iris_sc <- transform(sc, iris)
#'table(iris_sc$Species)
#'@export
sample_groups <- function(attribute, n_groups) {
  obj <- dal_transform()
  obj$attribute <- attribute
  obj$n_groups <- n_groups
  class(obj) <- append("sample_groups", class(obj))
  return(obj)
}

#'@exportS3Method transform sample_groups
transform.sample_groups <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  attribute <- obj$attribute
  if (is.null(attribute) || !attribute %in% names(data)) {
    stop("sample_groups: 'attribute' must be a valid column name in data.")
  }
  groups <- unique(data[[attribute]])
  if (length(groups) < obj$n_groups) {
    stop("sample_groups: n_groups exceeds the number of available groups.")
  }
  selected <- sample(groups, size = obj$n_groups, replace = FALSE)
  data[data[[attribute]] %in% selected, , drop=FALSE]
}
