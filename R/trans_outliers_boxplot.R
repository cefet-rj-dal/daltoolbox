#'@title outliers_boxplot
#'@description The outliers_boxplot class uses box-plot definition for outliers_boxplot.
#'An outlier is a value that is below than \eqn{Q_1 - 1.5 \cdot IQR} or higher than \eqn{Q_3 + 1.5 \cdot IQR}.
#'The class remove outliers_boxplot for numeric attributes.
#'Users can set alpha to 3 to remove extreme values.
#'@param alpha boxplot outlier threshold (default 1.5, but can be 3.0 to remove extreme values)
#'@return returns an outlier object
#'@examples
#'# code for outlier removal
#' out_obj <- outliers_boxplot() # class for outlier analysis
#' out_obj <- fit(out_obj, iris) # computing boundaries
#' iris.clean <- transform(out_obj, iris) # returning cleaned dataset
#'
#' #inspection of cleaned dataset
#' nrow(iris.clean)
#'
#' idx <- attr(iris.clean, "idx")
#' table(idx)
#' iris.outliers_boxplot <- iris[idx,]
#' iris.outliers_boxplot
#'@export
outliers_boxplot <- function(alpha = 1.5) {
  obj <- dal_transform()
  obj$alpha <- alpha
  class(obj) <- append("outliers_boxplot", class(obj))
  return(obj)
}

#'@importFrom stats quantile
#'@exportS3Method fit outliers_boxplot
fit.outliers_boxplot <- function(obj, data, ...) {
  lower_threshold <- NA
  higher_threshold <- NA
  if(is.matrix(data) || is.data.frame(data)) {
    lower_threshold <- rep(NA, ncol(data))
    higher_threshold <- rep(NA, ncol(data))
    if (nrow(data) >= 30) {
      for (i in 1:ncol(data)) {
        if (is.numeric(data[,i])) {
          # quartiles for IQR-based thresholds
          q <- stats::quantile(data[,i])
          IQR <- q[4] - q[2]
          lower_threshold[i] <- q[2] - obj$alpha*IQR
          higher_threshold[i] <- q[4] + obj$alpha*IQR
        }
      }
    }
  }
  else {
    if ((length(data) >= 30) && is.numeric(data)) {
      # vector input: same IQR logic
      q <- stats::quantile(data)
      IQR <- q[4] - q[2]
      lower_threshold <- q[2] - obj$alpha*IQR
      higher_threshold <- q[4] + obj$alpha*IQR
    }
  }
  obj$lower_threshold <- lower_threshold
  obj$higher_threshold <- higher_threshold
  return(obj)
}

#'@exportS3Method transform outliers_boxplot
transform.outliers_boxplot <- function(obj, data, ...) {
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
    data <- adjust_matrix(data[!idx,]) # keep only inlier rows
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
