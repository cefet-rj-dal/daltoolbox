#'@title cSPADE sequences
#'@description Sequential pattern mining using `arulesSequences::cspade`.
#'@param support minimum support threshold. If `0`, estimated during `fit()` using `support_strategy`.
#'@param support_strategy support threshold strategy created with `pat_support_threshold()`
#'@param maxsize maximum number of items per event
#'@param maxlen maximum number of events per sequence
#'@param mingap minimum gap between successive events
#'@param maxgap maximum gap between successive events
#'@param quality_filter optional quality filter created with `patutils()`
#'@param control list of control parameters
#'@return returns a `pat_cspade` object
#'@examples
#'if (requireNamespace("arulesSequences", quietly = TRUE)) {
#'  x <- arulesSequences::read_baskets(
#'    con = system.file("misc", "zaki.txt", package = "arulesSequences"),
#'    info = c("sequenceID", "eventID", "SIZE")
#'  )
#'  utils <- patutils()
#'  pm <- pat_cspade(
#'    support = 0,
#'    support_strategy = pat_support_threshold("min_count", min_count = 2),
#'    maxlen = 3,
#'    quality_filter = utils$quality_min(support = 0.5)
#'  )
#'  pm <- fit(pm, x)
#'  seqs <- discover(pm, x)
#'  eval <- evaluate(pm, seqs)
#'  eval$metrics
#'}
#'@export
pat_cspade <- function(support = 0,
                       support_strategy = pat_support_threshold("min_count", min_count = 2),
                       maxsize = NULL,
                       maxlen = NULL,
                       mingap = NULL,
                       maxgap = NULL,
                       quality_filter = NULL,
                       control = list(verbose = TRUE)) {
  obj <- pattern_miner()
  utils <- obj$pat_utils
  obj$support <- support
  obj$support_strategy <- support_strategy
  obj$maxsize <- maxsize
  obj$maxlen <- maxlen
  obj$mingap <- mingap
  obj$maxgap <- maxgap
  obj$quality_filter <- quality_filter
  obj$control <- control
  obj$pattern_kind <- "sequences"
  obj$eval_metrics <- list(
    utils$metric_pattern_count,
    utils$metric_mean_support,
    utils$metric_mean_length,
    utils$metric_retained_ratio
  )
  class(obj) <- append("pat_cspade", class(obj))
  return(obj)
}

pat_cspade_compile <- function(obj, data = NULL) {
  obj$support_resolved <- pat_resolve_support(
    obj[["support", exact = TRUE]],
    obj[["support_strategy", exact = TRUE]],
    data
  )
  obj$engine_parameter <- list(support = obj$support_resolved)
  if (!is.null(obj$maxsize)) obj$engine_parameter$maxsize <- obj$maxsize
  if (!is.null(obj$maxlen)) obj$engine_parameter$maxlen <- obj$maxlen
  if (!is.null(obj$mingap)) obj$engine_parameter$mingap <- obj$mingap
  if (!is.null(obj$maxgap)) obj$engine_parameter$maxgap <- obj$maxgap
  obj$engine_control <- obj$control
  obj
}

#'@exportS3Method fit pat_cspade
fit.pat_cspade <- function(obj, data, ...) {
  if (!requireNamespace("arulesSequences", quietly = TRUE)) {
    stop("pat_cspade requires the 'arulesSequences' package.", call. = FALSE)
  }
  data <- pattern_prepare_sequences(data)
  obj <- pat_cspade_compile(obj, data)
  pattern_miner_mark_fitted(obj, data)
}

#'@importFrom arulesSequences cspade
#'@exportS3Method discover pat_cspade
discover.pat_cspade <- function(obj, data, ...) {
  pattern_miner_require_fitted(obj)
  if (missing(data)) stop("pat_cspade: data is required.")
  if (!requireNamespace("arulesSequences", quietly = TRUE)) {
    stop("pat_cspade requires the 'arulesSequences' package.", call. = FALSE)
  }
  data <- pattern_prepare_sequences(data)
  validate_pattern_schema(obj, data)
  patterns <- arulesSequences::cspade(
    data,
    parameter = obj$engine_parameter,
    control = obj$engine_control,
    ...
  )
  patterns <- pattern_miner_apply_quality_filter(obj, patterns)
  patterns
}
