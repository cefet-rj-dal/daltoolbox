#'@title Apriori rules
#'@description Frequent itemsets and association rules using `arules::apriori`.
#'@param target mining target: `"rules"` or `"frequent itemsets"`
#'@param supp minimum support threshold. If `0`, estimated during `fit()` using `support_strategy`.
#'@param conf minimum confidence threshold for rules. If `0`, estimated during `fit()` using `confidence_strategy`.
#'@param support_strategy support threshold strategy created with `pat_support_threshold()`
#'@param confidence_strategy confidence threshold strategy created with `pat_confidence_threshold()`
#'@param minlen minimum pattern length
#'@param maxlen maximum pattern length
#'@param lhs optional vector of items constrained to the left-hand side of rules
#'@param rhs optional vector of items constrained to the right-hand side of rules
#'@param include optional vector of items allowed in the discovered patterns
#'@param exclude optional vector of items forbidden in the discovered patterns
#'@param quality_filter optional quality filter created with `patutils()`
#'@param rule_filter rule filter created with `pat_rule_filter_none()`, `pat_rule_filter_interest()`, or `pat_rule_filter_dara()`
#'@param control list of control parameters
#'@return returns a `pat_apriori` object
#'@examples
#'if (requireNamespace("arules", quietly = TRUE)) {
#'  data("AdultUCI", package = "arules")
#'  trans <- suppressWarnings(methods::as(as.data.frame(AdultUCI), "transactions"))
#'  utils <- patutils()
#'  pm <- pat_apriori(
#'    target = "rules",
#'    supp = 0,
#'    conf = 0,
#'    support_strategy = pat_support_threshold("curvature"),
#'    confidence_strategy = pat_confidence_threshold("rhs_baseline", margin = 0.1),
#'    minlen = 2,
#'    maxlen = 3,
#'    rhs = c("native-country=United-States"),
#'    quality_filter = utils$quality_min(confidence = 0.9, lift = 1.03),
#'    rule_filter = pat_rule_filter_interest(lift_min = 1),
#'    control = list(verbose = FALSE)
#'  )
#'  pm <- fit(pm, trans)
#'  rules <- suppressWarnings(discover(pm, trans))
#'  eval <- evaluate(pm, rules)
#'  eval$metrics
#'}
#'@export
pat_apriori <- function(target = c("rules", "frequent itemsets"),
                        supp = 0,
                        conf = 0,
                        support_strategy = pat_support_threshold("curvature"),
                        confidence_strategy = pat_confidence_threshold(),
                        minlen = 2,
                        maxlen = 10,
                        lhs = NULL,
                        rhs = NULL,
                        include = NULL,
                        exclude = NULL,
                        quality_filter = NULL,
                        rule_filter = pat_rule_filter_none(),
                        control = NULL) {
  target <- match.arg(target)
  obj <- pattern_miner()
  utils <- obj$pat_utils
  obj$target <- target
  obj$supp <- supp
  obj$conf <- conf
  obj$support_strategy <- support_strategy
  obj$confidence_strategy <- confidence_strategy
  obj$minlen <- minlen
  obj$maxlen <- maxlen
  obj$lhs <- lhs
  obj$rhs <- rhs
  obj$include <- include
  obj$exclude <- exclude
  obj$quality_filter <- quality_filter
  obj$rule_filter <- rule_filter
  obj$control <- control
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

pat_apriori_compile <- function(obj, data = NULL) {
  utils <- obj$pat_utils
  if (!is.null(obj$lhs) && obj$target != "rules") {
    stop("pat_apriori: 'lhs' is only valid when target = 'rules'.", call. = FALSE)
  }
  if (!is.null(obj$rhs) && obj$target != "rules") {
    stop("pat_apriori: 'rhs' is only valid when target = 'rules'.", call. = FALSE)
  }

  obj$supp_resolved <- pat_resolve_support(
    obj[["supp", exact = TRUE]],
    obj[["support_strategy", exact = TRUE]],
    data
  )
  obj$conf_resolved <- if (obj$target == "rules") {
    pat_resolve_confidence(
      obj[["conf", exact = TRUE]],
      obj[["confidence_strategy", exact = TRUE]],
      data,
      rhs = obj$rhs,
      support = obj$supp_resolved
    )
  } else {
    NULL
  }

  obj$engine_parameter <- list(
    supp = obj$supp_resolved,
    minlen = obj$minlen,
    maxlen = obj$maxlen,
    target = obj$target
  )
  if (obj$target == "rules") {
    obj$engine_parameter$conf <- obj$conf_resolved
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
  data <- pattern_prepare_transactions(data)
  obj <- pat_apriori_compile(obj, data)
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
  patterns <- pattern_miner_apply_rule_filter(obj, patterns, data = data)
  patterns
}
