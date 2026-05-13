#'@title Apriori rules
#'@description Frequent itemsets and association rules using `arules::apriori`.
#'@param target mining target: `"rules"` or `"frequent itemsets"`
#'@param supp minimum support threshold
#'@param conf minimum confidence threshold for rules
#'@param minlen minimum pattern length
#'@param maxlen maximum pattern length
#'@param lhs optional vector of items constrained to the left-hand side of rules
#'@param rhs optional vector of items constrained to the right-hand side of rules
#'@param include optional vector of items allowed in the discovered patterns
#'@param exclude optional vector of items forbidden in the discovered patterns
#'@param quality_filter optional quality filter created with `patutils()`
#'@param control list of control parameters
#'@param parameter legacy list of parameters passed to `arules::apriori`
#'@param appearance legacy list of item appearance constraints
#'@return returns a `pat_apriori` object
#'@examples
#'if (requireNamespace("arules", quietly = TRUE)) {
#'  data("AdultUCI", package = "arules")
#'  trans <- suppressWarnings(methods::as(as.data.frame(AdultUCI), "transactions"))
#'  utils <- patutils()
#'  pm <- pat_apriori(
#'    target = "rules",
#'    supp = 0.5,
#'    conf = 0.9,
#'    minlen = 2,
#'    maxlen = 4,
#'    rhs = c("native-country=United-States"),
#'    quality_filter = utils$quality_min(confidence = 0.95)
#'  )
#'  pm <- fit(pm, trans)
#'  rules <- discover(pm, trans)
#'  eval <- evaluate(pm, rules)
#'  eval$metrics
#'}
#'@export
pat_apriori <- function(target = c("rules", "frequent itemsets"),
                        supp = 0.5,
                        conf = 0.9,
                        minlen = 2,
                        maxlen = 10,
                        lhs = NULL,
                        rhs = NULL,
                        include = NULL,
                        exclude = NULL,
                        quality_filter = NULL,
                        control = NULL,
                        parameter = NULL,
                        appearance = NULL) {
  target <- match.arg(target)
  obj <- pattern_miner()
  utils <- obj$pat_utils
  obj$target <- target
  obj$supp <- supp
  obj$conf <- conf
  obj$minlen <- minlen
  obj$maxlen <- maxlen
  obj$lhs <- lhs
  obj$rhs <- rhs
  obj$include <- include
  obj$exclude <- exclude
  obj$quality_filter <- quality_filter
  obj$control <- control
  obj$parameter <- parameter
  obj$appearance <- appearance
  obj$pattern_kind <- if (target == "rules") "rules" else "itemsets"
  obj$eval_metrics <- list(
    utils$metric_pattern_count,
    utils$metric_mean_support,
    utils$metric_mean_confidence,
    utils$metric_mean_lift,
    utils$metric_mean_length,
    utils$metric_retained_ratio
  )
  class(obj) <- append("pat_apriori", class(obj))
  return(obj)
}

pat_apriori_compile <- function(obj) {
  utils <- obj$pat_utils
  if (!is.null(obj$parameter)) {
    legacy <- obj$parameter
    if (!is.null(legacy$supp)) obj$supp <- legacy$supp
    if (!is.null(legacy$conf)) obj$conf <- legacy$conf
    if (!is.null(legacy$minlen)) obj$minlen <- legacy$minlen
    if (!is.null(legacy$maxlen)) obj$maxlen <- legacy$maxlen
    if (!is.null(legacy$target)) obj$target <- legacy$target
    obj$pattern_kind <- if (obj$target == "rules") "rules" else "itemsets"
  }

  if (!is.null(obj$appearance)) {
    if (!is.null(obj$appearance$lhs)) obj$lhs <- obj$appearance$lhs
    if (!is.null(obj$appearance$rhs)) obj$rhs <- obj$appearance$rhs
  }

  if (!is.null(obj$lhs) && obj$target != "rules") {
    stop("pat_apriori: 'lhs' is only valid when target = 'rules'.", call. = FALSE)
  }
  if (!is.null(obj$rhs) && obj$target != "rules") {
    stop("pat_apriori: 'rhs' is only valid when target = 'rules'.", call. = FALSE)
  }

  obj$engine_parameter <- list(
    supp = obj$supp,
    minlen = obj$minlen,
    maxlen = obj$maxlen,
    target = obj$target
  )
  if (obj$target == "rules") {
    obj$engine_parameter$conf <- obj$conf
  }

  obj$engine_appearance <- NULL
  if (obj$target == "rules") {
    obj$eval_metrics <- list(
      utils$metric_pattern_count,
      utils$metric_mean_support,
      utils$metric_mean_confidence,
      utils$metric_mean_lift,
      utils$metric_mean_length,
      utils$metric_retained_ratio
    )
  } else {
    obj$eval_metrics <- list(
      utils$metric_pattern_count,
      utils$metric_mean_support,
      utils$metric_mean_length,
      utils$metric_retained_ratio
    )
  }

  if (!is.null(obj$lhs) && !is.null(obj$rhs)) {
    obj$engine_appearance <- list(lhs = obj$lhs, rhs = obj$rhs, default = "none")
  } else if (!is.null(obj$rhs)) {
    obj$engine_appearance <- list(rhs = obj$rhs, default = "lhs")
  } else if (!is.null(obj$lhs)) {
    obj$engine_appearance <- list(lhs = obj$lhs, default = "rhs")
  }

  obj$engine_control <- obj$control
  obj
}

#'@importFrom methods as
#'@exportS3Method fit pat_apriori
fit.pat_apriori <- function(obj, data, ...) {
  if (!requireNamespace("arules", quietly = TRUE)) {
    stop("pat_apriori requires the 'arules' package.", call. = FALSE)
  }
  obj <- pat_apriori_compile(obj)
  data <- pattern_prepare_transactions(data)
  pattern_miner_mark_fitted(obj, data)
}

#'@importFrom arules apriori
#'@importFrom methods as
#'@exportS3Method discover pat_apriori
discover.pat_apriori <- function(obj, data, ...) {
  pattern_miner_require_fitted(obj)
  if (missing(data)) stop("pat_apriori: data is required.")
  if (!requireNamespace("arules", quietly = TRUE)) {
    stop("pat_apriori requires the 'arules' package.", call. = FALSE)
  }
  data <- pattern_prepare_transactions(data)
  validate_pattern_schema(obj, data)
  patterns <- arules::apriori(
    data,
    parameter = obj$engine_parameter,
    appearance = obj$engine_appearance,
    control = obj$engine_control,
    ...
  )
  patterns <- pattern_miner_apply_item_filter(obj, patterns)
  patterns <- pattern_miner_apply_quality_filter(obj, patterns)
  patterns
}
