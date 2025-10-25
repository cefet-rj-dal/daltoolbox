#'@title PAM
#'@description Creates a clusterer object that
#' uses the Partition Around Medoids (PAM) method
#' It wraps the cluster library.
#'@param k the number of clusters to generate.
#'@return returns PAM object.
#'@examples
#'# setup clustering
#'model <- cluster_pam(k = 3)
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
cluster_pam <- function(k = 1) {
  obj <- clusterer()
  obj$k <- k

  class(obj) <- append("cluster_pam", class(obj))
  return(obj)
}

#'@importFrom cluster pam
#'@exportS3Method cluster cluster_pam
cluster.cluster_pam <- function(obj, data, ...) {
  pam_cluster <- cluster::pam(data, obj$k)
  cluster <- pam_cluster$cluster

  #intrinsic quality metric
  dist <- 0
  for (i in 1:obj$k) {
    idx <- i==pam_cluster$clustering
    center <- pam_cluster$medoids[i,]
    # sum of squared distances to medoids
    dist <- dist + sum(rowSums((data[idx,] - center)^2))
  }

  attr(cluster, "metric") <- dist
  return(cluster)
}

