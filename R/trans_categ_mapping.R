#'@title Categorical mapping (one‑hot encoding)
#'@description Convert a factor column into dummy variables (one‑hot encoding) using `model.matrix` without intercept.
#' Each level becomes a separate binary column named `<attribute>_<level>`.
#'@details The remaining columns are kept and the original attribute is replaced in place by
#' its one-hot dummy columns (appended at the end). Keeping the other columns makes the
#' transform safe to use inside a preprocessing pipeline.
#'@param attribute attribute to be categorized.
#'@return returns the input data frame with the attribute replaced by one binary column per
#' category (`<attribute>_<level>`); all other columns are preserved.
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
  attr <- obj$attribute
  data[[attr]] <- as.factor(data[[attr]])
  # one-hot via model.matrix without intercept, named "<attr>_<level>"
  mdlattribute <- stats::formula(paste("~", paste(attr, "-1")))
  dummies <- as.data.frame(stats::model.matrix(mdlattribute, data = data))
  colnames(dummies) <- paste(attr, levels(data[[attr]]), sep = "_")
  # keep the other columns (original order), replace the attribute by its dummies
  others <- data[, setdiff(names(data), attr), drop = FALSE]
  return(cbind(others, dummies))
}

