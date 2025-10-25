#'@title DBSCAN
#'@description Creates a clusterer object that
#' uses the DBSCAN method
#' It wraps the dbscan library.
#'@param eps distance value
#'@param minPts minimum number of points
#'@return returns a dbscan object
#'@examples
#'# setup clustering
#'model <- cluster_dbscan(minPts = 3)
#'
#'#load dataset
#'data(iris)
#'
#'# build model
#'model <- fit(model, iris[,1:4])
#'clu <- cluster(model, iris[,1:4])
#'table(clu)
#'
#'# evaluate model using external metric
#'eval <- evaluate(model, clu, iris$Species)
#'eval
#'@export
cluster_dbscan <- function(minPts = 3, eps = NULL) {
  obj <- clusterer()
  obj$minPts <- minPts
  obj$eps <- eps

  class(obj) <- append("cluster_dbscan", class(obj))
  return(obj)
}

#'@title fit dbscan model
#'@description Fits a DBSCAN clustering model by setting the `eps` parameter.
#'If `eps` is not provided, it is estimated based on the k-nearest neighbor distances.
#'It wraps dbscan library
#'@param obj an object containing the DBSCAN model configuration, including `minPts` and optionally `eps`
#'@param data the dataset to use for fitting the model
#'@param ... optional arguments
#'@return returns a fitted obj with the `eps` parameter set
#'@importFrom dbscan kNNdist
#'@exportS3Method fit cluster_dbscan
fit.cluster_dbscan <- function(obj, data, ...) {
  if (is.null(obj$eps)) {
    # sort k-NN distances and pick epsilon at elbow (max curvature)
    t <- sort(dbscan::kNNdist(data, k = obj$minPts))
    y <- t
    myfit <- fit_curvature_max()
    res <- transform(myfit, y)
    obj$eps <- res$y
  }
  return(obj)
}


#'@importFrom dbscan dbscan
#'@exportS3Method cluster cluster_dbscan
cluster.cluster_dbscan <- function(obj, data, ...) {
  db_cluster <- dbscan::dbscan(data, eps = obj$eps, minPts = obj$minPts)
  cluster <- db_cluster$cluster

  #intrinsic quality metric
  # number of noise points (cluster 0)
  null_cluster <- length(cluster[cluster==0])
  attr(cluster, "metric") <- null_cluster

  return(cluster)
}

