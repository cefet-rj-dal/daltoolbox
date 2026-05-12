#'@title Random class undersampling
#'@description Balance class distributions by randomly reducing all classes to the minority count.
#'@param attribute target class attribute name
#'@return returns an object of class `bal_subsampling`
#'@examples
#'data(iris)
#'iris_imb <- iris[c(1:50, 51:71, 101:111), ]
#'bal <- bal_subsampling("Species")
#'iris_bal <- transform(bal, iris_imb)
#'table(iris_bal$Species)
#'@export
bal_subsampling <- function(attribute) {
  obj <- dal_transform()
  obj$attribute <- attribute
  class(obj) <- append("bal_subsampling", class(obj))
  return(obj)
}

#'@exportS3Method transform bal_subsampling
transform.bal_subsampling <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  attribute <- obj$attribute
  if (is.null(attribute) || !attribute %in% names(data)) {
    stop("bal_subsampling: attribute not found in data.")
  }

  counts <- sort(table(data[[attribute]]))
  target_n <- as.integer(counts[1])
  classes <- names(counts)
  parts <- vector("list", length(classes))

  for (i in seq_along(classes)) {
    class_data <- data[data[[attribute]] == classes[i], , drop = FALSE]
    idx <- sample(seq_len(nrow(class_data)), size = target_n)
    parts[[i]] <- class_data[idx, , drop = FALSE]
  }

  result <- do.call(rbind, parts)
  rownames(result) <- NULL
  return(result)
}
