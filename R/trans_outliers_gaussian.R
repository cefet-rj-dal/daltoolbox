#'@title Outlier removal by Gaussian 3-sigma rule
#'@description Removes outliers from numeric columns using the 3‑sigma rule under a Gaussian assumption:
#' values outside mean ± alpha·sd are flagged as outliers.
#'@param alpha gaussian threshold (default 3)
#'@return returns an outlier object
#'@references
#' Pukelsheim, F. (1994). The Three Sigma Rule. The American Statistician 48(2):88–91.
#'@examples
#'# code for outlier removal
#' out_obj <- outliers_gaussian() # class for outlier analysis
#' out_obj <- fit(out_obj, iris) # computing boundaries
#' iris.clean <- transform(out_obj, iris) # returning cleaned dataset
#'
#' #inspection of cleaned dataset
#' nrow(iris.clean)
#'
#' idx <- attr(iris.clean, "idx")
#' table(idx)
#' iris.outliers_gaussian <- iris[idx,]
#' iris.outliers_gaussian
#'@export
outliers_gaussian <- function(alpha = 3) {
  obj <- dal_transform()
  obj$alpha <- alpha
  class(obj) <- append("outliers_gaussian", class(obj))
  return(obj)
}

#'@importFrom stats sd
#'@exportS3Method fit outliers_gaussian
fit.outliers_gaussian <- function(obj, data, ...) {
  lower_threshold <- NA
  higher_threshold <- NA
  if(is.matrix(data) || is.data.frame(data)) {
    lower_threshold <- rep(NA, ncol(data))
    higher_threshold <- rep(NA, ncol(data))
    if (nrow(data) >= 30) {
      for (i in 1:ncol(data)) {
        if (is.numeric(data[,i])) {
          # mean ± alpha * sd per numeric column
          q <- base::mean(data[,i])
          s <- stats::sd(data[,i])
          lower_threshold[i] <- q - obj$alpha*s
          higher_threshold[i] <-  q + obj$alpha*s
        }
      }
    }
  }
  else {
    if ((length(data) >= 30) && is.numeric(data)) {
      # vector input: mean ± alpha * sd
      q <- mean(data)
      s <- sd(data)
      lower_threshold <- q - obj$alpha*s
      higher_threshold <-  q + obj$alpha*s
    }
  }
  obj$lower_threshold <- lower_threshold
  obj$higher_threshold <- higher_threshold
  return(obj)
}

#'@exportS3Method transform outliers_gaussian
transform.outliers_gaussian <- function(obj, data, ...) {
  idx <- FALSE
  lower_threshold <- obj$lower_threshold
  higher_threshold <- obj$higher_threshold
  if (is.matrix(data) || is.data.frame(data)) {
    idx = rep(FALSE, nrow(data))
    for (i in 1:ncol(data))
      if (!is.na(lower_threshold[i]) && !is.na(higher_threshold[i]))
        idx = idx | (!is.na(data[,i]) & (data[,i] < lower_threshold[i] | data[,i] > higher_threshold[i]))
  }
  if(is.matrix(data))
    data <- adjust_matrix(data[!idx,])
  else if (is.data.frame(data))
    data <- adjust_data.frame(data[!idx,])
  else {
    if (!is.na(lower_threshold) && !is.na(higher_threshold)) {
      idx <- data < lower_threshold | data > higher_threshold
      data <- data[!idx]
    }
    else
      idx <- rep(FALSE, length(data))
  }
  attr(data, "idx") <- idx
  return(data)
}
