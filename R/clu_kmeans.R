#'@title k-means
#'@description k-means clustering using `stats::kmeans`.
#'@details Partitions data into k clusters minimizing within‑cluster sum of squares. The intrinsic
#' quality metric returned is the total within‑cluster SSE (lower is better).
#'@param k the number of clusters to form.
#'@return returns a k-means object.
#'@references
#' MacQueen, J. (1967). Some Methods for classification and Analysis of Multivariate Observations.
#' Lloyd, S. (1982). Least squares quantization in PCM.
#'@examples
#'# setup clustering
#'model <- cluster_kmeans(k=3)
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
cluster_kmeans <- function(k = 1) {
  obj <- clusterer()
  obj$k <- k
  class(obj) <- append("cluster_kmeans", class(obj))
  return(obj)
}

#'@importFrom stats kmeans
#'@exportS3Method cluster cluster_kmeans
cluster.cluster_kmeans <- function(obj, data, ...) {
  k <- obj$k
  k_cluster <- stats::kmeans(x = data, centers = k)
  cluster <- k_cluster$cluster

  #intrinsic quality metric
  dist <- 0
  for (i in 1:k) {
    idx <- i == k_cluster$cluster
    center <- k_cluster$centers[i,]
    # sum of squared distances within clusters
    dist <- dist + sum(rowSums((data[idx,] - center)^2))
  }
  attr(cluster, "metric") <- dist

  return(cluster)
}

