#'@title Categorical mapping
#'@description Categorical mapping provides a way to map the levels of a categorical variable to new values.
#' Each possible value is converted to a binary attribute.
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
#'@export
transform.categ_mapping <- function(obj, data, ...) {
  mdlattribute <- stats::formula(paste("~", paste(obj$attribute, "-1")))
  data <- as.data.frame(stats::model.matrix(mdlattribute, data=data))
  data[,obj$attribute] <- NULL
  return(data)
}

