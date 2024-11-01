#'@title k-means
#'@description Creates a clusterer object that
#' uses the k-means method
#' It wraps the stats library.
#'@param k the number of clusters to form.
#'@return returns a k-means object.
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
#'@export
cluster.cluster_kmeans <- function(obj, data, ...) {
  k <- obj$k
  k_cluster <- stats::kmeans(x = data, centers = k)
  cluster <- k_cluster$cluster

  #intrinsic quality metric
  dist <- 0
  for (i in 1:k) {
    idx <- i == k_cluster$cluster
    center <- k_cluster$centers[i,]
    dist <- dist + sum(rowSums((data[idx,] - center)^2))
  }
  attr(cluster, "metric") <- dist

  return(cluster)
}

