#'@title Min-max normalization
#'@description The minmax performs scales data between \[0,1\].
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
minmax <- function() {
  obj <- dal_transform()
  class(obj) <- append("minmax", class(obj))
  return(obj)
}

#'@exportS3Method fit minmax
fit.minmax <- function(obj, data, ...) {
  # create a small metadata frame to track which columns are numeric
  minmax = data.frame(t(ifelse(sapply(data, is.numeric), 1, 0)))
  # append rows to store max and min per numeric column
  minmax = rbind(minmax, rep(NA, ncol(minmax)))
  minmax = rbind(minmax, rep(NA, ncol(minmax)))
  colnames(minmax) = colnames(data)
  rownames(minmax) = c("numeric", "max", "min")
  # compute min/max only for numeric columns (skip factors/characters)
  for (j in colnames(minmax)[minmax["numeric",]==1]) {
    minmax["min",j] <- min(data[,j], na.rm=TRUE)
    minmax["max",j] <- max(data[,j], na.rm=TRUE)
  }
  obj$norm.set <- minmax
  return(obj)
}

#'@exportS3Method transform minmax
transform.minmax <- function(obj, data, ...) {
  minmax <- obj$norm.set
  # apply (x - min) / (max - min) per numeric column
  for (j in colnames(minmax)[minmax["numeric",]==1]) {
    if ((minmax["max", j] != minmax["min", j])) {
      data[,j] <- (data[,j] - minmax["min", j]) / (minmax["max", j] - minmax["min", j])
    }
    else {
      # when max == min, set constant columns to 0 to avoid division by zero
      data[,j] <- 0
    }
  }
  return (data)
}

#'@exportS3Method inverse_transform minmax
inverse_transform.minmax <- function(obj, data, ...) {
  minmax <- obj$norm.set
  # revert normalization per numeric column
  for (j in colnames(minmax)[minmax["numeric",]==1]) {
    if ((minmax["max", j] != minmax["min", j])) {
      data[,j] <- data[,j] * (minmax["max", j] - minmax["min", j]) + minmax["min", j]
    }
    else {
      # for constant columns, original value equals max (== min)
      data[,j] <- minmax["max", j]
    }
  }
  return (data)
}

