#'@title ts_data
#'@description Time series data structure used in DAL Toolbox.
#'It receives a vector (representing a time series) or
#'a matrix `y` (representing a sliding windows).
#'Internal ts_data is matrix of sliding windows with size `sw`.
#'If sw equals to zero, it store a time series as a single matrix column.
#'@param y output variable
#'@param sw integer: sliding window size.
#'@return returns a `ts_data` object.
#'@examples
#'data(sin_data)
#'head(sin_data)
#'
#'data <- ts_data(sin_data$y)
#'ts_head(data)
#'
#'data10 <- ts_data(sin_data$y, 10)
#'ts_head(data10)
#'@export
ts_data <- function(y, sw=1) {
  #https://stackoverflow.com/questions/7532845/matrix-losing-class-attribute-in-r
  ts_sw <- function(x, sw) {
    ts_lag <- function(x, k)
    {
      c(rep(NA, k), x)[1 : length(x)]
    }
    n <- length(x)-sw+1
    window <- NULL
    for(c in (sw-1):0){
      t  <- ts_lag(x,c)
      t <- t[sw:length(t)]
      window <- cbind(window,t,deparse.level = 0)
    }
    col <- paste("t",c((sw-1):0), sep="")
    colnames(window) <- col
    return(window)
  }

  if (sw > 1)
    y <- ts_sw(as.matrix(y), sw)
  else {
    y <- as.matrix(y)
    sw <- 1
  }

  col <- paste("t",(ncol(y)-1):0, sep="")
  colnames(y) <- col

  class(y) <- append("ts_data", class(y))
  attr(y, "sw") <- sw
  return(y)
}

#'@title Subset Extraction for Time Series Data
#'@description Extracts a subset of a time series object based on specified rows and columns.
#'The function allows for flexible indexing and subsetting of time series data.
#'@param x `ts_data` object
#'@param i row i
#'@param j column j
#'@param ... optional arguments
#'@return returns a new ts_data object
#'@examples
#'data(sin_data)
#'data10 <- ts_data(sin_data$y, 10)
#'ts_head(data10)
#'#single line
#'data10[12,]
#'
#'#range of lines
#'data10[12:13,]
#'
#'#single column
#'data10[,1]
#'
#'#range of columns
#'data10[,1:2]
#'
#'#range of rows and columns
#'data10[12:13,1:2]
#'
#'#single line and a range of columns
#'#'data10[12,1:2]
#'
#'#range of lines and a single column
#'data10[12:13,1]
#'
#'#single observation
#'data10[12,1]
#'@export
`[.ts_data` <- function(x, i, j, ...) {
  y <- unclass(x)[i, j, drop = FALSE, ...]
  class(y) <- append("ts_data", class(y))
  attr(y, "sw") <- ncol(y)
  return(y)
}

#'@title Extract the First Observations from a `ts_data` Object
#'@description Returns the first n observations from a `ts_data`
#'@param x `ts_data` object
#'@param n number of rows to return
#'@param ... optional arguments
#'@return returns the first n observations of a `ts_data`
#'@examples
#'data(sin_data)
#'data10 <- ts_data(sin_data$y, 10)
#'ts_head(data10)
#'@importFrom utils head
#'@export
ts_head <- function(x, n = 6L, ...) {
  utils::head(unclass(x), n)
}

