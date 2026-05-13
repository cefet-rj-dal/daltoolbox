#'@title Pattern mining utilities
#'@description Utility object that groups filtering helpers and evaluation metrics for pattern mining.
#'@return returns a `patutils` object
#'@examples
#'utils <- patutils()
#'utils$quality_min(confidence = 0.8, lift = 1.1)
#'@export
patutils <- function() {
  metric_result <- function(metric, value, type = "intrinsic", details = NULL) {
    result <- list(metric = metric, value = as.numeric(value), type = type)
    if (!is.null(details)) {
      result$details <- details
    }
    result
  }

  quality_min <- function(support = NULL, confidence = NULL, lift = NULL) {
    list(min = Filter(Negate(is.null), list(
      support = support,
      confidence = confidence,
      lift = lift
    )))
  }

  quality_max <- function(support = NULL, confidence = NULL, lift = NULL) {
    list(max = Filter(Negate(is.null), list(
      support = support,
      confidence = confidence,
      lift = lift
    )))
  }

  item_tokens <- function(labels) {
    lapply(labels, function(label) {
      tokens <- unlist(regmatches(label, gregexpr("[^,{}<> ]+", label)))
      tokens <- tokens[tokens != "" & tokens != "=>"]
      unique(tokens)
    })
  }

  pattern_quality <- function(patterns) {
    if (!requireNamespace("arules", quietly = TRUE)) {
      return(data.frame())
    }
    quality <- arules::quality(patterns)
    if (is.null(quality)) {
      return(data.frame())
    }
    as.data.frame(quality)
  }

  pattern_length <- function(patterns) {
    if (!requireNamespace("arules", quietly = TRUE)) {
      return(rep(NA_real_, length(patterns)))
    }
    out <- tryCatch(arules::size(patterns), error = function(cond) rep(NA_real_, length(patterns)))
    as.numeric(out)
  }

  metric_pattern_count <- function(patterns, obj = NULL, ...) {
    metric_result("pattern_count", length(patterns), "intrinsic")
  }

  metric_mean_support <- function(patterns, obj = NULL, ...) {
    quality <- pattern_quality(patterns)
    value <- if ("support" %in% colnames(quality)) mean(quality$support, na.rm = TRUE) else NA_real_
    metric_result("mean_support", value, "intrinsic")
  }

  metric_mean_confidence <- function(patterns, obj = NULL, ...) {
    quality <- pattern_quality(patterns)
    value <- if ("confidence" %in% colnames(quality)) mean(quality$confidence, na.rm = TRUE) else NA_real_
    metric_result("mean_confidence", value, "intrinsic")
  }

  metric_mean_lift <- function(patterns, obj = NULL, ...) {
    quality <- pattern_quality(patterns)
    value <- if ("lift" %in% colnames(quality)) mean(quality$lift, na.rm = TRUE) else NA_real_
    metric_result("mean_lift", value, "intrinsic")
  }

  metric_mean_length <- function(patterns, obj = NULL, ...) {
    lengths <- pattern_length(patterns)
    value <- if (all(is.na(lengths))) NA_real_ else mean(lengths, na.rm = TRUE)
    metric_result("mean_length", value, "intrinsic")
  }

  metric_retained_ratio <- function(patterns, obj = NULL, ...) {
    total <- attr(patterns, "filtered_from", exact = TRUE)
    if (is.null(total) || is.na(total) || total == 0) {
      return(metric_result("retained_ratio", NA_real_, "filter"))
    }
    metric_result("retained_ratio", length(patterns) / total, "filter")
  }

  obj <- dal_base()
  class(obj) <- append("patutils", class(obj))
  obj$metric_result <- metric_result
  obj$quality_min <- quality_min
  obj$quality_max <- quality_max
  obj$item_tokens <- item_tokens
  obj$pattern_quality <- pattern_quality
  obj$pattern_length <- pattern_length
  obj$metric_pattern_count <- metric_pattern_count
  obj$metric_mean_support <- metric_mean_support
  obj$metric_mean_confidence <- metric_mean_confidence
  obj$metric_mean_lift <- metric_mean_lift
  obj$metric_mean_length <- metric_mean_length
  obj$metric_retained_ratio <- metric_retained_ratio
  return(obj)
}
