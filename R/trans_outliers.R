#'@title Outliers
#'@description The outliers class uses box-plot definition for outliers.
#'An outlier is a value that is below than \eqn{Q_1 - 1.5 \cdot IQR} or higher than \eqn{Q_3 + 1.5 \cdot IQR}.
#'The class remove outliers for numeric attributes.
#'Users can set alpha to 3 to remove extreme values.
#'@param alpha boxplot outlier threshold (default 1.5, but can be 3.0 to remove extreme values)
#'@return returns an outlier object
#'@examples
#'# code for outlier removal
#' out_obj <- outliers() # class for outlier analysis
#' out_obj <- fit(out_obj, iris) # computing boundaries
#' iris.clean <- transform(out_obj, iris) # returning cleaned dataset
#'
#' #inspection of cleaned dataset
#' nrow(iris.clean)
#'
#' idx <- attr(iris.clean, "idx")
#' table(idx)
#' iris.outliers <- iris[idx,]
#' iris.outliers
#'@export
outliers <- function(alpha = 1.5) {
  obj <- dal_transform()
  obj$alpha <- alpha
  class(obj) <- append("outliers", class(obj))
  return(obj)
}

#'@importFrom stats quantile
#'@exportS3Method fit outliers
fit.outliers <- function(obj, data, ...) {
  lq1 <- NA
  hq3 <- NA
  if(is.matrix(data) || is.data.frame(data)) {
    lq1 <- rep(NA, ncol(data))
    hq3 <- rep(NA, ncol(data))
    if (nrow(data) >= 30) {
      for (i in 1:ncol(data)) {
        if (is.numeric(data[,i])) {
          q <- stats::quantile(data[,i])
          IQR <- q[4] - q[2]
          lq1[i] <- q[2] - obj$alpha*IQR
          hq3[i] <- q[4] + obj$alpha*IQR
        }
      }
    }
  }
  else {
    if ((length(data) >= 30) && is.numeric(data)) {
      q <- stats::quantile(data)
      IQR <- q[4] - q[2]
      lq1 <- q[2] - obj$alpha*IQR
      hq3 <- q[4] + obj$alpha*IQR
    }
  }
  obj$lq1 <- lq1
  obj$hq3 <- hq3
  return(obj)
}

#'@exportS3Method transform outliers
transform.outliers <- function(obj, data, ...) {
  idx <- FALSE
  lq1 <- obj$lq1
  hq3 <- obj$hq3
  if (is.matrix(data) || is.data.frame(data)) {
    idx = rep(FALSE, nrow(data))
    for (i in 1:ncol(data))
      if (!is.na(lq1[i]) && !is.na(hq3[i]))
        idx = idx | (!is.na(data[,i]) & (data[,i] < lq1[i] | data[,i] > hq3[i]))
  }
  if(is.matrix(data))
    data <- adjust_matrix(data[!idx,])
  else if (is.data.frame(data))
    data <- adjust_data.frame(data[!idx,])
  else {
    if (!is.na(lq1) && !is.na(hq3)) {
      idx <- data < lq1 | data > hq3
      data <- data[!idx]
    }
    else
      idx <- rep(FALSE, length(data))
  }
  attr(data, "idx") <- idx
  return(data)
}
