#'@title Cluster sampling
#'@description Sample entire clusters defined by a categorical attribute.
#'@param attribute cluster attribute name
#'@param n_clusters number of clusters to sample
#'@param seed optional random seed for reproducibility
#'@return returns an object of class `sample_cluster`
#'@examples
#'data(iris)
#'sc <- sample_cluster("Species", n_clusters = 2, seed = 123)
#'iris_sc <- transform(sc, iris)
#'table(iris_sc$Species)
#'@export
sample_cluster <- function(attribute, n_clusters, seed = NULL) {
  obj <- dal_transform()
  obj$attribute <- attribute
  obj$n_clusters <- n_clusters
  obj$seed <- seed
  class(obj) <- append("sample_cluster", class(obj))
  return(obj)
}

#'@exportS3Method transform sample_cluster
transform.sample_cluster <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  attribute <- obj$attribute
  if (is.null(attribute) || !attribute %in% names(data)) {
    stop("sample_cluster: 'attribute' must be a valid column name in data.")
  }
  clusters <- unique(data[[attribute]])
  if (length(clusters) < obj$n_clusters) {
    stop("sample_cluster: n_clusters exceeds the number of available clusters.")
  }
  if (!is.null(obj$seed)) {
    set.seed(obj$seed)
  }
  selected <- sample(clusters, size = obj$n_clusters, replace = FALSE)
  data[data[[attribute]] %in% selected, , drop=FALSE]
}
