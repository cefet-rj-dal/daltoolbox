#'@title PCA
#'@description PCA (Principal Component Analysis) is an unsupervised
#' dimensionality reduction technique used in data analysis and
#' machine learning. It transforms a dataset of possibly
#' correlated variables into a new set of uncorrelated
#' variables called principal components.
#'@param attribute target attribute to model building
#'@param components number of components for PCA
#'@return returns an object of class `dt_pca`
#'@examples
#'mypca <- dt_pca("Species")
#'# Automatically fitting number of components
#'mypca <- fit(mypca, iris)
#'iris.pca <- transform(mypca, iris)
#'head(iris.pca)
#'head(mypca$pca.transf)
#'# Manual establishment of number of components
#'mypca <- dt_pca("Species", 3)
#'mypca <- fit(mypca, datasets::iris)
#'iris.pca <- transform(mypca, iris)
#'head(iris.pca)
#'head(mypca$pca.transf)
#'@export
dt_pca <- function(attribute=NULL, components = NULL) {
  obj <- dal_transform()
  obj$attribute <- attribute
  obj$components <- components
  class(obj) <- append("dt_pca", class(obj))
  return(obj)
}

#'@importFrom stats prcomp
#'@exportS3Method fit dt_pca
fit.dt_pca <- function(obj, data, ...) {
  data <- data.frame(data)
  attribute <- obj$attribute
  if (!is.null(attribute)) {
    # drop target column from PCA input (unsupervised)
    data[,attribute] <- NULL
  }
  # select numeric columns only
  nums <- unlist(lapply(data, is.numeric))
  remove <- NULL
  for(j in names(nums[nums])) {
    # remove constant columns (zero variance)
    if(min(data[,j])==max(data[,j]))
      remove <- cbind(remove, j)
  }
  nums[remove] <- FALSE
  data = as.matrix(data[ , nums])

  pca_res <- stats::prcomp(data, center=TRUE, scale.=TRUE)

  if (is.null(obj$components)) {
    # choose number of components via elbow (minimum curvature of cumulative variance)
    y <-  cumsum(pca_res$sdev^2/sum(pca_res$sdev^2))
    curv <-  fit_curvature_min()
    res <- transform(curv, y)
    obj$components <- res$x

  }

  obj$pca.transf <- as.matrix(pca_res$rotation[, 1:obj$components])
  obj$nums <- nums

  return(obj)
}

#'@exportS3Method transform dt_pca
transform.dt_pca <- function(obj, data, ...) {
  attribute <- obj$attribute
  pca.transf <- obj$pca.transf
  nums <- obj$nums

  data <- data.frame(data)
  if (!is.null(attribute)) {
    # preserve predictand and remove from PCA input
    predictand <- data[,attribute]
    data[,attribute] <- NULL
  }
  data = as.matrix(data[ , nums])

  # project to principal components
  data = data %*% pca.transf
  data = data.frame(data)
  if (!is.null(attribute)){
    # reattach predictand
    data[,attribute] <- predictand
  }
  return(data)
}
