#' @title No rule filtering
#' @description Build an explicit rule filter that leaves discovered rules unchanged.
#' @details
#' Use this filter when the miner should only discover candidate rules and skip
#' post-mining interestingness filtering. It is the default rule filter for
#' association-rule miners.
#' @return A `pat_rule_filter_none` object.
#' @examples
#' filter <- pat_rule_filter_none()
#' class(filter)
#' @export
pat_rule_filter_none <- function() {
  obj <- dal_base()
  class(obj) <- append(c("pat_rule_filter_none", "pat_rule_filter"), class(obj))
  obj
}

#' @title Interesting rule filter
#' @description Build a rule filter based on classical interest measures.
#' @details
#' This filter follows the usual post-processing step for association rules:
#' rules are discovered first, then kept only when their interest measures pass
#' user-defined thresholds. `lift > 1`, high Kulczynski, and low imbalance ratio
#' are common criteria discussed in association-rule mining texts such as Han,
#' Kamber, and Pei.
#' @param lift_min Minimum lift. Use `NULL` to ignore lift.
#' @param kulc_min Minimum Kulczynski measure. Use `NULL` to ignore Kulczynski.
#' @param ir_max Maximum imbalance ratio. Use `NULL` to ignore imbalance ratio.
#' @param support_min Minimum support. Use `NULL` to ignore support.
#' @param confidence_min Minimum confidence. Use `NULL` to ignore confidence.
#' @param count_min Minimum rule count. Use `NULL` to ignore count.
#' @return A `pat_rule_filter_interest` object.
#' @examples
#' filter <- pat_rule_filter_interest(lift_min = 1, kulc_min = 0.6, ir_max = 0.9)
#' class(filter)
#' @export
pat_rule_filter_interest <- function(lift_min = 1,
                                     kulc_min = NULL,
                                     ir_max = NULL,
                                     support_min = NULL,
                                     confidence_min = NULL,
                                     count_min = NULL) {
  obj <- dal_base()
  obj$criteria <- Filter(Negate(is.null), list(
    lift_min = lift_min,
    kulc_min = kulc_min,
    ir_max = ir_max,
    support_min = support_min,
    confidence_min = confidence_min,
    count_min = count_min
  ))
  class(obj) <- append(c("pat_rule_filter_interest", "pat_rule_filter"), class(obj))
  obj
}

#' @title DARA rule filter
#' @description Build a rule filter based on Divergent Association Ranking Analysis.
#' @details
#' DARA filters rules by scoring each antecedent with the divergence counters
#' computed by `pat_dara()`. A rule with antecedent values that are more prominent
#' in the rule set than in the target subset receives a higher positive score.
#' @param data Reference data frame used by DARA.
#' @param rhs_attr Target attribute used to restrict `data`.
#' @param rhs_value Optional target value used with `rhs_attr`.
#' @param attributes Optional attributes to compare. Defaults to common columns.
#' @param min_score Minimum DARA score required to keep a rule.
#' @param top Optional number of highest-scored rules to keep.
#' @param score Rule score aggregation: `"sum"`, `"mean"`, or `"max"`.
#' @param use_abs If `TRUE`, rank by absolute divergence counters.
#' @param rule_weight Optional numeric rule column used to weight rule frequencies.
#' @return A `pat_rule_filter_dara` object.
#' @examples
#' data <- data.frame(a = factor(c("x", "x", "y")), b = factor(c("z", "w", "z")))
#' filter <- pat_rule_filter_dara(data, rhs_attr = "b", rhs_value = "z")
#' class(filter)
#' @export
pat_rule_filter_dara <- function(data,
                                 rhs_attr,
                                 rhs_value = NULL,
                                 attributes = NULL,
                                 min_score = 1,
                                 top = NULL,
                                 score = c("sum", "mean", "max"),
                                 use_abs = FALSE,
                                 rule_weight = NULL) {
  score <- match.arg(score)
  obj <- dal_base()
  obj$data <- adjust_data.frame(data)
  obj$rhs_attr <- rhs_attr
  obj$rhs_value <- rhs_value
  obj$attributes <- attributes
  obj$min_score <- min_score
  obj$top <- top
  obj$score <- score
  obj$use_abs <- use_abs
  obj$rule_weight <- rule_weight
  class(obj) <- append(c("pat_rule_filter_dara", "pat_rule_filter"), class(obj))
  obj
}

#' @title Apply a rule filter
#' @description Filter association rules using a none, interest, or DARA rule filter.
#' @param rules An `arules` rules object or a tidy rule data frame.
#' @param filter A rule filter produced by `pat_rule_filter_none()`, `pat_rule_filter_interest()`, or `pat_rule_filter_dara()`.
#' @param data Optional transaction/reference data used to compute interest measures for `arules` rules.
#' @param ... Additional arguments reserved for future filters.
#' @return Filtered rules in the same representation as `rules`.
#' @examples
#' if (requireNamespace("arules", quietly = TRUE)) {
#'   data("AdultUCI", package = "arules")
#'   data <- as.data.frame(AdultUCI)
#'   data <- data[, c("workclass", "education", "marital-status", "occupation", "income")]
#'   rules <- pat_dara_rules(data, rhs = "income=small", supp = 0.05, conf = 0.6)
#'   filter <- pat_rule_filter_interest(lift_min = 1)
#'   pat_filter_rules(rules, filter)
#' }
#' @export
pat_filter_rules <- function(rules, filter, data = NULL, ...) {
  if (!inherits(filter, "pat_rule_filter")) {
    stop("pat_filter_rules: 'filter' must be a pat_rule_filter object.", call. = FALSE)
  }
  if (inherits(filter, "pat_rule_filter_none")) {
    return(pat_filter_rules_none(rules, filter, data = data, ...))
  }
  if (inherits(filter, "pat_rule_filter_interest")) {
    return(pat_filter_rules_interest(rules, filter, data = data, ...))
  }
  if (inherits(filter, "pat_rule_filter_dara")) {
    return(pat_filter_rules_dara(rules, filter, data = data, ...))
  }
  stop("pat_filter_rules: unsupported rule filter.", call. = FALSE)
}

pat_filter_rules_none <- function(rules, filter, data = NULL, ...) {
  if (inherits(rules, "rules")) {
    attr(rules, "filtered_from") <- length(rules)
    return(rules)
  }
  if (is.data.frame(rules)) {
    attr(rules, "filtered_from") <- nrow(rules)
    return(rules)
  }
  stop("pat_filter_rules expects an 'arules' rules object or a tidy rule data frame.", call. = FALSE)
}

pat_filter_rules_interest <- function(rules, filter, data = NULL, ...) {
  prepared <- pat_rule_filter_prepare_rules(rules, data)
  tidy <- prepared$tidy
  idx <- rep(TRUE, nrow(tidy))
  criteria <- filter$criteria

  idx <- pat_rule_filter_threshold(idx, tidy, "lift", criteria$lift_min, ">=")
  idx <- pat_rule_filter_threshold(idx, tidy, "kulc", criteria$kulc_min, ">=")
  idx <- pat_rule_filter_threshold(idx, tidy, "ir", criteria$ir_max, "<=")
  idx <- pat_rule_filter_threshold(idx, tidy, "support", criteria$support_min, ">=")
  idx <- pat_rule_filter_threshold(idx, tidy, "confidence", criteria$confidence_min, ">=")
  idx <- pat_rule_filter_threshold(idx, tidy, "count", criteria$count_min, ">=")

  pat_rule_filter_restore(prepared, idx, score = NULL)
}

pat_filter_rules_dara <- function(rules, filter, data = NULL, ...) {
  reference_data <- if (is.data.frame(data) || is.matrix(data)) data else filter$data
  prepared <- pat_rule_filter_prepare_rules(rules, reference_data)
  tidy <- prepared$tidy
  if (nrow(tidy) == 0) {
    return(pat_rule_filter_restore(prepared, logical()))
  }

  ranking <- pat_dara(
    filter$data,
    tidy,
    rhs_attr = filter$rhs_attr,
    rhs_value = filter$rhs_value,
    attributes = filter$attributes,
    rule_weight = filter$rule_weight
  )
  scores <- pat_rule_filter_dara_scores(tidy, ranking, filter$score, filter$use_abs)
  idx <- scores >= filter$min_score
  if (!is.null(filter$top) && length(scores) > 0) {
    ord <- order(scores, decreasing = TRUE)
    keep <- rep(FALSE, length(scores))
    keep[head(ord, filter$top)] <- TRUE
    idx <- idx & keep
  }

  pat_rule_filter_restore(prepared, idx, score = scores, ranking = ranking)
}

pat_rule_filter_prepare_rules <- function(rules, data = NULL) {
  if (inherits(rules, "rules")) {
    item_columns <- if (is.data.frame(data) || is.matrix(data)) colnames(data) else NULL
    tidy <- pat_rules_tidy(rules, transactions = data, item_columns = item_columns)
    return(list(original = rules, tidy = tidy, arules = TRUE))
  }
  if (is.data.frame(rules)) {
    return(list(original = rules, tidy = adjust_data.frame(rules), arules = FALSE))
  }
  stop("pat_filter_rules expects an 'arules' rules object or a tidy rule data frame.", call. = FALSE)
}

pat_rule_filter_restore <- function(prepared, idx, score = NULL, ranking = NULL) {
  if (length(idx) == 0) {
    idx <- rep(FALSE, nrow(prepared$tidy))
  }
  idx[is.na(idx)] <- FALSE
  if (isTRUE(prepared$arules)) {
    out <- prepared$original[idx]
  } else {
    out <- prepared$original[idx, , drop = FALSE]
    if (!is.null(score)) {
      out$dara_score <- score[idx]
    }
  }
  if (!is.null(score)) {
    attr(out, "dara_score") <- score[idx]
  }
  if (!is.null(ranking)) {
    attr(out, "dara_ranking") <- ranking
  }
  attr(out, "filtered_from") <- nrow(prepared$tidy)
  out
}

pat_rule_filter_threshold <- function(idx, rules, column, threshold, op) {
  if (is.null(threshold)) {
    return(idx)
  }
  if (!column %in% colnames(rules)) {
    stop(sprintf("pat_rule_filter_interest: rule column '%s' is required.", column), call. = FALSE)
  }
  values <- as.numeric(rules[[column]])
  if (op == ">=") {
    return(idx & values >= threshold)
  }
  if (op == "<=") {
    return(idx & values <= threshold)
  }
  idx
}

pat_rule_filter_dara_scores <- function(rules, ranking, score, use_abs) {
  scores <- numeric(nrow(rules))
  for (i in seq_len(nrow(rules))) {
    parsed <- pat_dara_parse_items(rules$lhs[[i]])
    vals <- numeric()
    for (attribute in names(parsed)) {
      if (!attribute %in% names(ranking)) {
        next
      }
      table <- ranking[[attribute]]
      hit <- table$c[table$value == unname(parsed[[attribute]])]
      if (length(hit) > 0) {
        vals <- c(vals, hit[[1]])
      }
    }
    if (use_abs) {
      vals <- abs(vals)
    }
    scores[[i]] <- switch(
      score,
      sum = sum(vals),
      mean = if (length(vals) == 0) 0 else mean(vals),
      max = if (length(vals) == 0) 0 else max(vals)
    )
  }
  scores
}

pattern_miner_apply_rule_filter <- function(obj, patterns, data = NULL) {
  rule_filter <- obj$rule_filter
  if (is.null(rule_filter)) {
    rule_filter <- pat_rule_filter_none()
  }
  if (!inherits(patterns, "rules")) {
    stop("pattern_miner: rule_filter can only be applied to association rules.", call. = FALSE)
  }
  pat_filter_rules(patterns, rule_filter, data = data)
}
