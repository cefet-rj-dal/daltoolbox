#'@title Simple sampling
#'@description Sample rows or elements with or without replacement.
#'@param size number of samples to draw
#'@param replace logical; sample with replacement if TRUE
#'@param prob optional vector of sampling probabilities
#'@param seed optional random seed for reproducibility
#'@return returns an object of class `sample_simple`
#'@examples
#'data(iris)
#'srswor <- sample_simple(size = 10, replace = FALSE, seed = 123)
#'srswr <- sample_simple(size = 10, replace = TRUE, seed = 123)
#'sample_wor <- transform(srswor, iris$Sepal.Length)
#'sample_wr <- transform(srswr, iris$Sepal.Length)
#'sample_wor
#'sample_wr
#'@export
sample_simple <- function(size, replace = FALSE, prob = NULL, seed = NULL) {
  obj <- dal_transform()
  obj$size <- size
  obj$replace <- replace
  obj$prob <- prob
  obj$seed <- seed
  class(obj) <- append("sample_simple", class(obj))
  return(obj)
}

#'@exportS3Method transform sample_simple
transform.sample_simple <- function(obj, data, ...) {
  if (!is.null(obj$seed)) {
    set.seed(obj$seed)
  }
  size <- obj$size
  replace <- obj$replace
  prob <- obj$prob

  if (is.matrix(data) || is.data.frame(data)) {
    data <- adjust_data.frame(data)
    idx <- sample(seq_len(nrow(data)), size = size, replace = replace, prob = prob)
    data <- data[idx, , drop=FALSE]
    return(data)
  }

  sample(data, size = size, replace = replace, prob = prob)
}
