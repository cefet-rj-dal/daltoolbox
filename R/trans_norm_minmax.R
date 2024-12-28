#'@title Min-max normalization
#'@description The minmax performs scales data between \[0,1\].
#'@param minmax_list list(feat=list(min=x, max=y)...) List of lists with min and max for each feature.
#'
#'\eqn{minmax = (x-min(x))/(max(x)-min(x))}
#'@return returns an object of class `minmax`
#'@examples
#'data(iris)
#'head(iris)
#'
#'trans <- minmax()
#'trans <- fit(trans, iris)
#'tiris <- transform(trans, iris)
#'head(tiris)
#'
#'itiris <- inverse_transform(trans, tiris)
#'head(itiris)
#'@export
minmax <- function(minmax_list=NULL) {
  obj <- dal_transform()
  obj$minmax_list <- minmax_list
  class(obj) <- append("minmax", class(obj))
  return(obj)
}

#'@export
fit.minmax <- function(obj, data, ...) {
  minmax = data.frame(t(ifelse(sapply(data, is.numeric), 1, 0)))
  minmax = rbind(minmax, rep(NA, ncol(minmax)))
  minmax = rbind(minmax, rep(NA, ncol(minmax)))
  colnames(minmax) = colnames(data)
  rownames(minmax) = c("numeric", "max", "min")
  for (j in colnames(minmax)[minmax["numeric",]==1]) {
    if(!is.null(obj$minmax_list)){
      minmax["min",j] <- obj$minmax_list[[j]]$min
      minmax["max",j] <- obj$minmax_list[[j]]$max
    }else{
      minmax["min",j] <- min(data[,j], na.rm=TRUE)
      minmax["max",j] <- max(data[,j], na.rm=TRUE)
    }
    
  }
  obj$norm.set <- minmax
  return(obj)
}


#'@export
transform.minmax <- function(obj, data, ...) {
  minmax <- obj$norm.set
  for (j in colnames(minmax)[minmax["numeric",]==1]) {
    if ((minmax["max", j] != minmax["min", j])) {
      data[,j] <- (data[,j] - minmax["min", j]) / (minmax["max", j] - minmax["min", j])
    }
    else {
      data[,j] <- 0
    }
  }
  return (data)
}

#'@export
inverse_transform.minmax <- function(obj, data, ...) {
  minmax <- obj$norm.set
  for (j in colnames(minmax)[minmax["numeric",]==1]) {
    if ((minmax["max", j] != minmax["min", j])) {
      data[,j] <- data[,j] * (minmax["max", j] - minmax["min", j]) + minmax["min", j]
    }
    else {
      data[,j] <- minmax["max", j]
    }
  }
  return (data)
}

