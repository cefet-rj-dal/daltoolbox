#'@title Adjust to matrix
#'@description Coerce an object to `matrix` if needed (useful before algorithms that expect matrices).
#'@param data dataset
#'@return returns an adjusted matrix
#'@examples
#'data(iris)
#'mat <- adjust_matrix(iris)
#'@export
adjust_matrix <- function(data) {
  if(!is.matrix(data)) {
    # coerce data.frame/vectors to matrix
    return(as.matrix(data))
  }
  else
    return(data)
}

#'@title  Adjust to data frame
#'@description Coerce an object to `data.frame` if needed (useful for S3 methods in this package).
#'@param data dataset
#'@return returns a data.frame
#'@examples
#'data(iris)
#'df <- adjust_data.frame(iris)
#'@export
adjust_data.frame <- function(data) {
  if(!is.data.frame(data)) {
    # coerce matrix/vectors to data.frame
    return(as.data.frame(data))
  }
  else
    return(data)
}

#'@title Adjust factors
#'@description Convert a vector to a factor with specified internal levels (`ilevels`) and labels (`slevels`).
#'@details Numeric vectors are first converted to factors with `ilevels` as the level order, then relabeled to `slevels`.
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
#'@description One‑hot encode a factor vector into a matrix of indicator columns.
#'@details Values are mapped to `valTrue`/`valFalse` (default 1/0). The resulting matrix has column names equal to levels(x).
#'@param x vector to be categorized
#'@param valTrue value to represent true
#'@param valFalse value to represent false
#'@return returns an adjusted categorical mapping
#'@export
adjust_class_label <- function (x, valTrue = 1, valFalse = 0)
{
  n <- length(x)
  x <- as.factor(x)
  # one-hot encode factor levels into a matrix
  res <- matrix(valFalse, n, length(levels(x)))
  res[(1:n) + n * (unclass(x) - 1)] <- valTrue
  dimnames(res) <- list(names(x), levels(x))
  res
}

