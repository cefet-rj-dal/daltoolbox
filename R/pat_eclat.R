#'@title ECLAT itemsets
#'@description Frequent itemsets using `arules::eclat`.
#'@param parameter list of parameters passed to `arules::eclat`
#'@param control list of control parameters
#'@return returns a `pat_eclat` object
#'@examples
#'data("AdultUCI", package = "arules")
#'trans <- suppressWarnings(methods::as(as.data.frame(AdultUCI), "transactions"))
#'pm <- pat_eclat(parameter = list(supp = 0.5, maxlen = 3))
#'pm <- fit(pm, trans)
#'itemsets <- discover(pm)
#'arules::inspect(itemsets[1:6])
#'@export
pat_eclat <- function(parameter = list(supp = 0.5, maxlen = 3),
                      control = NULL) {
  obj <- pattern_miner()
  obj$parameter <- parameter
  obj$control <- control
  class(obj) <- append("pat_eclat", class(obj))
  return(obj)
}

#'@importFrom arules eclat
#'@importFrom methods as
#'@exportS3Method discover pat_eclat
discover.pat_eclat <- function(obj, data = NULL, ...) {
  if (is.null(data)) data <- obj$data
  if (is.null(data)) stop("pat_eclat: data is required.")
  if (!inherits(data, "transactions")) {
    data <- methods::as(data, "transactions")
  }
  arules::eclat(
    data,
    parameter = obj$parameter,
    control = obj$control,
    ...
  )
}
