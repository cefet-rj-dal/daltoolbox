#'@title Adjust to matrix
#'@description Converts a dataset to a matrix format if it is not already in that format
#'@param data dataset
#'@return returns an adjusted matrix
#'@examples
#'data(iris)
#'mat <- adjust_matrix(iris)
#'@export
adjust_matrix <- function(data) {
  if(!is.matrix(data)) {
    return(as.matrix(data))
  }
  else
    return(data)
}

#'@title  Adjust to data frame
#'@description Converts a dataset to a `data.frame` if it is not already in that format
#'@param data dataset
#'@return returns a data.frame
#'@examples
#'data(iris)
#'df <- adjust_data.frame(iris)
#'@export
adjust_data.frame <- function(data) {
  if(!is.data.frame(data)) {
    return(as.data.frame(data))
  }
  else
    return(data)
}

#'@title Adjust factors
#'@description Converts a vector into a factor with specified levels and labels
#'@param value vector to be converted into factor
#'@param ilevels order for categorical values
#'@param slevels labels for categorical values
#'@return returns an adjusted factor
#'@export
adjust_factor <- function(value, ilevels, slevels) {
  if (!is.factor(value)) {
    if (is.numeric(value))
      value <- factor(value, levels=ilevels)
    levels(value) <- slevels
  }
  return(value)
}

#'@title Adjust categorical mapping
#'@description Converts a vector into a categorical mapping, where each category is represented by a specific value.
#'By default, the values represent binary categories (true/false)
#'@param x vector to be categorized
#'@param valTrue value to represent true
#'@param valFalse value to represent false
#'@return returns an adjusted categorical mapping
#'@export
adjust_class_label <- function (x, valTrue = 1, valFalse = 0)
{
  n <- length(x)
  x <- as.factor(x)
  res <- matrix(valFalse, n, length(levels(x)))
  res[(1:n) + n * (unclass(x) - 1)] <- valTrue
  dimnames(res) <- list(names(x), levels(x))
  res
}

