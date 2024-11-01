#'@title Smoothing
#'@description Smoothing is a statistical technique used to reduce
#' the noise in a signal or a dataset by removing the high-frequency components.
#' The smoothing level is associated with the number of bins used.
#' There are alternative methods to establish the smoothing:
#' equal interval, equal frequency, and clustering.
#'@param n number of bins
#'@return returns an object of class `smoothing`
#'@examples
#'data(iris)
#'obj <- smoothing_inter(n = 2)
#'obj <- fit(obj, iris$Sepal.Length)
#'sl.bi <- transform(obj, iris$Sepal.Length)
#'table(sl.bi)
#'obj$interval
#'
#'entro <- evaluate(obj, as.factor(names(sl.bi)), iris$Species)
#'entro$entropy
#'@export
smoothing <- function(n) {
  obj <- dal_transform()
  obj$n <- n
  obj$tune <- function(obj, data) {
    options <- obj$n
    opt <- data.frame()
    interval <- list()
    for (i in options)
    {
      obj$n <- i
      obj <- fit(obj, data)
      vm <- transform(obj, data)
      mse <- mean((data - vm)^2, na.rm = TRUE)
      row <- c(mse , i)
      opt <- rbind(opt, row)
    }
    colnames(opt)<-c("mean","num")
    curv <- fit_curvature_max()
    res <- transform(curv, opt$mean)
    obj$n <- res$x
    return(obj)
  }
  class(obj) <- append("smoothing", class(obj))
  return(obj)
}

#'@export
fit.smoothing <- function(obj, data, ...) {
  v <- data
  interval <- obj$interval
  names(interval) <- NULL
  interval[1] <- min(v)
  interval[length(interval)] <- max(v)
  interval.adj <- interval
  interval.adj[1] <- -.Machine$double.xmax
  interval.adj[length(interval)] <- .Machine$double.xmax
  obj$interval <- interval
  obj$interval.adj <- interval.adj
  return(obj)
}

#'@export
transform.smoothing <- function(obj, data, ...) {
  v <- data
  interval.adj <- obj$interval.adj
  vp <- cut(v, unique(interval.adj), FALSE, include.lowest=TRUE)
  m <- tapply(v, vp, mean)
  vm <- m[vp]
  return(vm)
}

#'@export
evaluate.smoothing <- function(obj, data, attribute, ...) {
  x <- y <- q <- qtd <- e <- n <- 0
  result <- list(data=as.factor(data), attribute=as.factor(attribute))

  compute_entropy <- function(obj) {
    value <- getOption("dplyr.summarise.inform")
    options(dplyr.summarise.inform = FALSE)

    dataset <- data.frame(x = obj$data, y = obj$attribute)
    tbl <- dataset |> dplyr::group_by(x, y) |> summarise(qtd=dplyr::n())
    tbs <- dataset |> dplyr::group_by(x) |> summarise(t=dplyr::n())
    tbl <- base::merge(x=tbl, y=tbs, by.x="x", by.y="x")
    tbl$e <- -(tbl$qtd/tbl$t)*log(tbl$qtd/tbl$t,2)
    tbl <- tbl |> dplyr::group_by(x) |> dplyr::summarise(ce=sum(e), qtd=sum(qtd))
    tbl$ceg <- tbl$ce*tbl$qtd/length(obj$data)
    obj$entropy_clusters <- tbl
    obj$entropy <- sum(obj$entropy$ceg)

    options(dplyr.summarise.inform = value)
    return(obj)
  }
  result <- compute_entropy(result)
  return(result)
}
