#'@title Missing value removal
#'@description Remove rows (or elements) that contain missing values.
#'@details For data frames or matrices, removes rows with any NA. For vectors, removes NA values.
#'@return returns an object of class `na_removal`
#'@examples
#'data(iris)
#'iris.na <- iris
#'iris.na$Sepal.Length[2] <- NA
#'obj <- na_removal()
#'iris.clean <- transform(obj, iris.na)
#'nrow(iris.clean)
#'@export
na_removal <- function() {
  obj <- dal_transform()
  class(obj) <- append("na_removal", class(obj))
  return(obj)
}

#'@exportS3Method transform na_removal
transform.na_removal <- function(obj, data, ...) {
  idx <- FALSE
  if (is.matrix(data) || is.data.frame(data)) {
    data <- adjust_data.frame(data)
    idx <- apply(data, 1, function(x) any(is.na(x)))
    data <- data[!idx, , drop=FALSE]
  } else {
    idx <- is.na(data)
    data <- data[!idx]
  }
  attr(data, "idx") <- idx
  return(data)
}
