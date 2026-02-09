#'@title cSPADE sequences
#'@description Sequential pattern mining using `arulesSequences::cspade`.
#'@param parameter list of parameters passed to `arulesSequences::cspade`
#'@param control list of control parameters
#'@return returns a `pat_cspade` object
#'@examples
#'x <- arulesSequences::read_baskets(
#'  con = system.file("misc", "zaki.txt", package = "arulesSequences"),
#'  info = c("sequenceID", "eventID", "SIZE")
#')
#'pm <- pat_cspade(parameter = list(support = 0.4))
#'pm <- fit(pm, x)
#'seqs <- discover(pm)
#'as(seqs, "data.frame")
#'@export
pat_cspade <- function(parameter = list(support = 0.4),
                       control = list(verbose = TRUE)) {
  obj <- pattern_miner()
  obj$parameter <- parameter
  obj$control <- control
  class(obj) <- append("pat_cspade", class(obj))
  return(obj)
}

#'@importFrom arulesSequences cspade
#'@exportS3Method discover pat_cspade
discover.pat_cspade <- function(obj, data = NULL, ...) {
  if (is.null(data)) data <- obj$data
  if (is.null(data)) stop("pat_cspade: data is required.")
  if (!inherits(data, "transactions")) {
    stop("pat_cspade: data must be a 'transactions' object (from arulesSequences::read_baskets).")
  }
  arulesSequences::cspade(
    data,
    parameter = obj$parameter,
    control = obj$control,
    ...
  )
}
