#'@title PCA
#'@description Principal Component Analysis (PCA) for unsupervised dimensionality reduction.
#' Transforms correlated variables into orthogonal principal components ordered by explained variance.
#'@details Fits PCA on (optionally) the numeric predictors only (excluding `attribute` when provided),
#' removes constant columns, and selects the number of components by an elbow rule (minimum curvature)
#' unless `components` is set explicitly. New data are projected with the same centering
#' and scaling learned during `fit()`.
#'@param attribute target attribute to model building
#'@param components number of components for PCA
#'@return returns an object of class `dt_pca`
#'@references
#' Pearson, K. (1901). On lines and planes of closest fit to systems of points in space.
#' Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components.
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
  feature_names <- names(nums[nums])
  data <- as.matrix(data[, feature_names, drop = FALSE])

  pca_res <- stats::prcomp(data, center=TRUE, scale.=TRUE)

  if (is.null(obj$components)) {
    # choose number of components via elbow (minimum curvature of cumulative variance)
    y <-  cumsum(pca_res$sdev^2/sum(pca_res$sdev^2))
    curv <-  fit_curvature_min()
    res <- transform(curv, y)
    obj$components <- res$x

  }

  obj$pca <- pca_res
  obj$pca.transf <- as.matrix(pca_res$rotation[, 1:obj$components, drop = FALSE])
  obj$feature_names <- feature_names

  return(obj)
}

#'@exportS3Method transform dt_pca
transform.dt_pca <- function(obj, data, ...) {
  attribute <- obj$attribute
  pca <- obj$pca
  feature_names <- obj$feature_names

  data <- data.frame(data)
  if (!is.null(attribute)) {
    # preserve predictand and remove from PCA input
    predictand <- data[,attribute]
    data[,attribute] <- NULL
  }
  missing_features <- setdiff(feature_names, names(data))
  if (length(missing_features) > 0) {
    stop(paste0(
      "dt_pca: missing numeric predictor columns in transform data: ",
      paste(missing_features, collapse = ", ")
    ))
  }
  data <- data[, feature_names, drop = FALSE]

  # project with the training centering/scaling learned by prcomp
  data <- stats::predict(pca, newdata = data)[, seq_len(obj$components), drop = FALSE]
  data <- data.frame(data)
  if (!is.null(attribute)){
    # reattach predictand
    data[,attribute] <- predictand
  }
  return(data)
}
