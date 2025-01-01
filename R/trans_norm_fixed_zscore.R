#'@title Fixed Z-score normalization
#'@description Scale data using z-score and minmax normalization.
#'
#'\eqn{zscore = (x - mean(x))/sd(x)}
#'@param nmean new mean for normalized data
#'@param nsd new standard deviation for normalized data
#'@return returns the z-score transformation object
#'@examples
#'data(iris)
#'head(iris)
#'
#'trans <- zscore()
#'trans <- fit(trans, iris)
#'tiris <- transform(trans, iris)
#'head(tiris)
#'
#'itiris <- inverse_transform(trans, tiris)
#'head(itiris)
#'@export
fixed_zscore <- function(nmean=0, nsd=1) {
  obj <- dal_transform()
  class(obj) <- append("fixed_zscore", class(obj))
  return(obj)
}


#'@importFrom stats sd
#'@export
fit.fixed_zscore <- function(obj, data, ...) {
  
  minmax_list <- list()
  for(col in names(data)){
    minmax_list[[col]] <- list(min=-5, max=5)
  }
  
  obj$zmodel <- zscore()
  obj$minmax_model <- minmax(minmax_list=minmax_list)
  obj$zmodel <- fit(obj$zmodel, data)
  z <- transform(obj$zmodel, data)
  obj$minmax_model <- fit(obj$minmax_model, z)
  
  return(obj)
}

#'@export
transform.fixed_zscore <- function(obj, data, ...) {
  
  z <- transform(obj$zmodel, data)
  data <- transform(obj$minmax_model, z)
  
  return(data)
}

#'@export
inverse_transform.fixed_zscore <- function(obj, data, ...) {

  z <- inverse_transform(obj$minmax_model, data)
  data <- inverse_transform(obj$minmax_model, z)
  
  return(data)
}
