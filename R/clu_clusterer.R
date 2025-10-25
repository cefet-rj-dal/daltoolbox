#'@title Clusterer
#'@description Ancestor class for clustering problems
#'@return returns a `clusterer` object
#'@examples
#'#See ?cluster_kmeans for an example of transformation
#'@export
clusterer <- function() {
  obj <- dal_learner()
  class(obj) <- append("clusterer", class(obj))
  return(obj)
}

#'@exportS3Method action clusterer
action.clusterer <- function(obj, ...) {
  thiscall <- match.call(expand.dots = TRUE)
  # proxy action() to cluster() for clusterers
  thiscall[[1]] <- as.name("cluster")
  result <- eval.parent(thiscall)
  return(result)
}

#'@title Cluster
#'@description Defines a cluster method
#'@param obj a `clusterer` object
#'@param ... optional arguments
#'@return clustered data
#'@examples
#'#See ?cluster_kmeans for an example of transformation
#' @export
cluster <- function(obj, ...) {
  UseMethod("cluster")
}

#'@exportS3Method cluster default
cluster.default <- function(obj, ...) {
  return(data.frame())
}

#'@importFrom dplyr filter summarise group_by n
#'@exportS3Method evaluate clusterer
evaluate.clusterer <- function(obj, cluster, attribute, ...) {
  compute_entropy <- function(obj) {
    x <- y <- e <- qtd <- n <- 0
    value <- getOption("dplyr.summarise.inform")
    options(dplyr.summarise.inform = FALSE)

    dataset <- data.frame(x = obj$data, y = obj$attribute)
    tbl <- dataset |> dplyr::group_by(x, y) |> dplyr::summarise(qtd=dplyr::n())
    tbs <- dataset |> dplyr::group_by(x) |> dplyr::summarise(t=dplyr::n())
    tbl <- base::merge(x=tbl, y=tbs, by.x="x", by.y="x")
    # per-cluster entropy contribution
    tbl$e <- -(tbl$qtd/tbl$t)*log(tbl$qtd/tbl$t,2)
    tbl <- tbl |> dplyr::group_by(x) |> dplyr::summarise(ce=sum(e), qtd=sum(qtd))
    # global entropy weighted by cluster sizes
    tbl$ceg <- tbl$ce*tbl$qtd/length(obj$data)

    options(dplyr.summarise.inform = value)

    result <- list()
    result$clusters_entropy <- tbl
    result$clustering_entropy <- sum(tbl$ceg)

    return(result)
  }

  # baseline entropy with a single cluster (upper bound)
  basic <- compute_entropy(list(data=as.factor(rep(1, length(attribute))), attribute=as.factor(attribute)))

  # actual clustering entropy
  result <- compute_entropy(list(data=as.factor(cluster), attribute=as.factor(attribute)))

  result$data_entropy <- basic$clustering_entropy

  return(result)
}
