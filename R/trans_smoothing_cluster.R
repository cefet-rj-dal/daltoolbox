#'@title Smoothing by class-aware clustering
#'@description Discretize a numeric attribute into `n` bins by clustering the attribute together
#' with a one-hot representation of the class label, then projecting the clusters back to
#' ordered cut points on the numeric axis.
#'@param class_label name of the class attribute
#'@param n number of bins
#'@return returns an object of class `smoothing_cluster`
#'@references
#' Han, J., Kamber, M., Pei, J. (2011). Data Mining: Concepts and Techniques. (Discretization)
#'@examples
#'data(iris)
#'cluster_data <- iris[, c("Sepal.Length", "Species")]
#'obj <- smoothing_cluster("Species", n = 2)
#'obj <- fit(obj, cluster_data)
#'sl.bi <- transform(obj, iris$Sepal.Length)
#'table(sl.bi)
#'obj$interval
#'
#'bins <- cut(iris$Sepal.Length, unique(obj$interval.adj), FALSE, include.lowest = TRUE)
#'entro <- evaluate(obj, bins, iris$Species)
#'entro$entropy
#'@export
smoothing_cluster <- function(class_label, n) {
  obj <- smoothing(n)
  obj$class_label <- class_label
  class(obj) <- append("smoothing_cluster", class(obj))
  return(obj)
}

smoothing_cluster_prepare <- function(obj, data) {
  data <- adjust_data.frame(data)
  if (!is.data.frame(data)) {
    stop("smoothing_cluster: data must be a data.frame with one numeric attribute and the class column.", call. = FALSE)
  }
  if (is.null(obj$class_label) || !obj$class_label %in% names(data)) {
    stop("smoothing_cluster: 'class_label' must be a valid column name in data.", call. = FALSE)
  }
  feature_names <- setdiff(names(data), obj$class_label)
  numeric_features <- feature_names[vapply(data[feature_names], is.numeric, logical(1))]
  if (length(numeric_features) != 1) {
    stop("smoothing_cluster: data must contain exactly one numeric attribute besides the class column.", call. = FALSE)
  }
  values <- data[[numeric_features]]
  if (anyNA(values)) {
    stop("smoothing_cluster: numeric attribute must not contain missing values.", call. = FALSE)
  }
  labels <- as.factor(data[[obj$class_label]])
  if (nlevels(labels) < 2) {
    stop("smoothing_cluster: class column must contain at least two levels.", call. = FALSE)
  }
  list(
    values = values,
    labels = labels,
    feature = numeric_features
  )
}

smoothing_cluster_fit_once <- function(obj, data) {
  prepared <- smoothing_cluster_prepare(obj, data)
  values <- prepared$values
  labels <- prepared$labels
  n <- obj$n
  if (length(unique(values)) < n) {
    stop("smoothing_cluster: n exceeds the number of distinct values in the numeric attribute.", call. = FALSE)
  }

  scaled_values <- if (stats::sd(values) == 0) rep(0, length(values)) else as.numeric(scale(values))
  class_matrix <- stats::model.matrix(~ labels - 1)
  km_input <- cbind(value = scaled_values, class_matrix)
  km <- stats::kmeans(x = km_input, centers = n)

  centers <- tapply(values, km$cluster, mean)
  centers <- sort(unname(centers))
  cuts <- if (length(centers) > 1) (centers[-length(centers)] + centers[-1]) / 2 else numeric(0)

  obj$feature <- prepared$feature
  obj$interval <- c(min(values), cuts, max(values))
  obj$cluster_centers <- centers
  fit.smoothing(obj, values)
}

#'@importFrom stats kmeans
#'@importFrom stats model.matrix
#'@exportS3Method fit smoothing_cluster
fit.smoothing_cluster <- function(obj, data, ...) {
  prepared <- smoothing_cluster_prepare(obj, data)
  if (length(obj$n) > 1) {
    options <- obj$n
    opt <- data.frame()
    for (i in options) {
      candidate <- obj
      candidate$n <- i
      candidate <- smoothing_cluster_fit_once(candidate, data)
      bins <- cut(prepared$values, unique(candidate$interval.adj), FALSE, include.lowest = TRUE)
      entropy <- evaluate(candidate, bins, prepared$labels)$entropy
      opt <- rbind(opt, c(entropy, i))
    }
    colnames(opt) <- c("entropy", "num")
    obj$optimization <- opt
    obj$n <- opt$num[which.min(opt$entropy)]
  }
  smoothing_cluster_fit_once(obj, data)
}
