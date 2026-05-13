#'@title ECLAT itemsets
#'@description Frequent itemsets using `arules::eclat`.
#'@param supp minimum support threshold
#'@param minlen minimum itemset length
#'@param maxlen maximum itemset length
#'@param include optional vector of items allowed in the discovered itemsets
#'@param exclude optional vector of items forbidden in the discovered itemsets
#'@param quality_filter optional quality filter created with `patutils()`
#'@param control list of control parameters
#'@param parameter legacy list of parameters passed to `arules::eclat`
#'@return returns a `pat_eclat` object
#'@examples
#'if (requireNamespace("arules", quietly = TRUE)) {
#'  data("AdultUCI", package = "arules")
#'  trans <- suppressWarnings(methods::as(as.data.frame(AdultUCI), "transactions"))
#'  utils <- patutils()
#'  pm <- pat_eclat(
#'    supp = 0.5,
#'    maxlen = 3,
#'    include = c("sex=Male", "income=small"),
#'    quality_filter = utils$quality_min(support = 0.55)
#'  )
#'  pm <- fit(pm, trans)
#'  itemsets <- discover(pm, trans)
#'  eval <- evaluate(pm, itemsets)
#'  eval$metrics
#'}
#'@export
pat_eclat <- function(supp = 0.5,
                      minlen = 1,
                      maxlen = 3,
                      include = NULL,
                      exclude = NULL,
                      quality_filter = NULL,
                      control = NULL,
                      parameter = NULL) {
  obj <- pattern_miner()
  utils <- obj$pat_utils
  obj$supp <- supp
  obj$minlen <- minlen
  obj$maxlen <- maxlen
  obj$include <- include
  obj$exclude <- exclude
  obj$quality_filter <- quality_filter
  obj$control <- control
  obj$parameter <- parameter
  obj$pattern_kind <- "itemsets"
  obj$eval_metrics <- list(
    utils$metric_pattern_count,
    utils$metric_mean_support,
    utils$metric_mean_length,
    utils$metric_retained_ratio
  )
  class(obj) <- append("pat_eclat", class(obj))
  return(obj)
}

pat_eclat_compile <- function(obj) {
  if (!is.null(obj$parameter)) {
    legacy <- obj$parameter
    if (!is.null(legacy$supp)) obj$supp <- legacy$supp
    if (!is.null(legacy$minlen)) obj$minlen <- legacy$minlen
    if (!is.null(legacy$maxlen)) obj$maxlen <- legacy$maxlen
  }

  obj$engine_parameter <- list(
    supp = obj$supp,
    minlen = obj$minlen,
    maxlen = obj$maxlen
  )
  obj$engine_control <- obj$control
  obj
}

#'@importFrom methods as
#'@exportS3Method fit pat_eclat
fit.pat_eclat <- function(obj, data, ...) {
  if (!requireNamespace("arules", quietly = TRUE)) {
    stop("pat_eclat requires the 'arules' package.", call. = FALSE)
  }
  obj <- pat_eclat_compile(obj)
  data <- pattern_prepare_transactions(data)
  pattern_miner_mark_fitted(obj, data)
}

#'@importFrom arules eclat
#'@importFrom methods as
#'@exportS3Method discover pat_eclat
discover.pat_eclat <- function(obj, data, ...) {
  pattern_miner_require_fitted(obj)
  if (missing(data)) stop("pat_eclat: data is required.")
  if (!requireNamespace("arules", quietly = TRUE)) {
    stop("pat_eclat requires the 'arules' package.", call. = FALSE)
  }
  data <- pattern_prepare_transactions(data)
  validate_pattern_schema(obj, data)
  patterns <- arules::eclat(
    data,
    parameter = obj$engine_parameter,
    control = obj$engine_control,
    ...
  )
  patterns <- pattern_miner_apply_item_filter(obj, patterns)
  patterns <- pattern_miner_apply_quality_filter(obj, patterns)
  patterns
}
