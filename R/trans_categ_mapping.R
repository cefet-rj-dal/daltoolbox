#'@title Categorical mapping (one‑hot encoding)
#'@description Convert a factor column into dummy variables (one‑hot encoding) using `model.matrix` without intercept.
#' Each level becomes a separate binary column.
#'@details This is a light wrapper around `stats::model.matrix(~ attr - 1, data)` that drops the original column
#' and returns only the dummy variables.
#'@param attribute attribute to be categorized.
#'@return returns a data frame with binary attributes, one for each possible category.
#'@examples
#'cm <- categ_mapping("Species")
#'iris_cm <- transform(cm, iris)
#'
#'# can be made in a single column
#'species <- iris[,"Species", drop=FALSE]
#'iris_cm <- transform(cm, species)
#'@export
categ_mapping <- function(attribute) {
  obj <- dal_transform()
  obj$attribute <- attribute
  class(obj) <- append("categ_mapping", class(obj))
  return(obj)
}

#'@importFrom stats formula
#'@importFrom stats model.matrix
#'@exportS3Method transform categ_mapping
transform.categ_mapping <- function(obj, data, ...) {
  # build formula without intercept for one-hot encoding
  mdlattribute <- stats::formula(paste("~", paste(obj$attribute, "-1")))
  data <- as.data.frame(stats::model.matrix(mdlattribute, data=data))
  data[,obj$attribute] <- NULL
  return(data)
}

