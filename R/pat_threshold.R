#' @title Pattern support threshold strategy
#' @description Configure how a minimum support threshold is estimated during `fit()`.
#' @param method Strategy name: `"curvature"`, `"min_count"`, `"sqrt"`, `"quantile"`, or `"density"`.
#' @param min_count Minimum absolute transaction count for `"min_count"`.
#' @param q Quantile of item frequencies for `"quantile"`.
#' @param scale Multiplicative factor used by `"quantile"` and `"density"`.
#' @param min Minimum allowed support.
#' @param max Maximum allowed support.
#' @return A `pat_support_threshold` object.
#' @examples
#' pat_support_threshold("curvature")
#' pat_support_threshold("min_count", min_count = 20)
#' pat_support_threshold("quantile", q = 0.25, scale = 0.5)
#' @export
pat_support_threshold <- function(method = c("curvature", "min_count", "sqrt", "quantile", "density"),
                                  min_count = 10,
                                  q = 0.25,
                                  scale = 1,
                                  min = 0.001,
                                  max = 0.5) {
  method <- match.arg(method)
  obj <- dal_base()
  obj$method <- method
  obj$min_count <- min_count
  obj$q <- q
  obj$scale <- scale
  obj$min <- min
  obj$max <- max
  class(obj) <- append("pat_support_threshold", class(obj))
  obj
}

#' @title Pattern confidence threshold strategy
#' @description Configure how a minimum confidence threshold is estimated during `fit()`.
#' @param method Strategy name: `"fixed"`, `"rhs_baseline"`, or `"support_adaptive"`.
#' @param value Fixed confidence used by `"fixed"`.
#' @param margin Added to the RHS baseline confidence for `"rhs_baseline"`.
#' @param scale Multiplicative factor used by `"support_adaptive"`.
#' @param min Minimum allowed confidence.
#' @param max Maximum allowed confidence.
#' @return A `pat_confidence_threshold` object.
#' @examples
#' pat_confidence_threshold("rhs_baseline", margin = 0.1, min = 0.5)
#' pat_confidence_threshold("support_adaptive", min = 0.6, max = 0.9)
#' @export
pat_confidence_threshold <- function(method = c("rhs_baseline", "support_adaptive", "fixed"),
                                     value = 0.8,
                                     margin = 0.1,
                                     scale = 1,
                                     min = 0.5,
                                     max = 0.95) {
  method <- match.arg(method)
  obj <- dal_base()
  obj$method <- method
  obj$value <- value
  obj$margin <- margin
  obj$scale <- scale
  obj$min <- min
  obj$max <- max
  class(obj) <- append("pat_confidence_threshold", class(obj))
  obj
}

pat_resolve_support <- function(value, strategy, data) {
  if (!is.null(value) && !identical(as.numeric(value)[[1]], 0)) {
    return(pat_clamp_threshold(value, 0, 1, "support"))
  }
  if (is.null(strategy)) {
    strategy <- pat_support_threshold()
  }
  if (!inherits(strategy, "pat_support_threshold")) {
    stop("pattern threshold: support strategy must be created with pat_support_threshold().", call. = FALSE)
  }

  n <- pattern_transaction_count(data)
  if (n <= 0) {
    stop("pattern threshold: support cannot be estimated from empty data.", call. = FALSE)
  }

  support <- switch(
    strategy$method,
    curvature = pattern_support_curvature(data),
    min_count = strategy$min_count / n,
    sqrt = sqrt(n) / n,
    quantile = {
      freq <- pattern_item_frequency(data)
      if (length(freq) == 0) strategy$min else as.numeric(stats::quantile(freq, probs = strategy$q, na.rm = TRUE)) * strategy$scale
    },
    density = {
      freq <- pattern_item_frequency(data)
      if (length(freq) == 0) strategy$min else mean(freq, na.rm = TRUE) * strategy$scale
    }
  )

  pat_clamp_threshold(support, strategy$min, strategy$max, "support")
}

pat_resolve_confidence <- function(value, strategy, data, rhs = NULL, support = NULL) {
  if (!is.null(value) && !identical(as.numeric(value)[[1]], 0)) {
    return(pat_clamp_threshold(value, 0, 1, "confidence"))
  }
  if (is.null(strategy)) {
    strategy <- pat_confidence_threshold()
  }
  if (!inherits(strategy, "pat_confidence_threshold")) {
    stop("pattern threshold: confidence strategy must be created with pat_confidence_threshold().", call. = FALSE)
  }

  confidence <- switch(
    strategy$method,
    fixed = strategy$value,
    rhs_baseline = pattern_rhs_baseline(data, rhs) + strategy$margin,
    support_adaptive = (1 - if (is.null(support)) 0 else support) * strategy$scale
  )

  pat_clamp_threshold(confidence, strategy$min, strategy$max, "confidence")
}

pat_clamp_threshold <- function(value, lower, upper, name) {
  value <- as.numeric(value)[[1]]
  if (is.na(value) || !is.finite(value)) {
    stop(sprintf("pattern threshold: %s must be finite.", name), call. = FALSE)
  }
  value <- base::max(lower, base::min(upper, value))
  value
}

pattern_transaction_count <- function(data) {
  if (inherits(data, "transactions")) {
    return(length(data))
  }
  if (is.data.frame(data) || is.matrix(data)) {
    return(nrow(data))
  }
  length(data)
}

pattern_item_frequency <- function(data) {
  if (inherits(data, "transactions")) {
    return(arules::itemFrequency(data))
  }
  if (is.data.frame(data) || is.matrix(data)) {
    trans <- pattern_prepare_transactions(data)
    return(arules::itemFrequency(trans))
  }
  numeric()
}

pattern_support_curvature <- function(data) {
  freq <- sort(pattern_item_frequency(data), decreasing = TRUE)
  freq <- freq[is.finite(freq) & !is.na(freq) & freq > 0]
  if (length(freq) == 0) {
    return(0.001)
  }
  if (length(freq) < 4) {
    return(stats::median(freq))
  }
  curv <- fit_curvature_max()
  res <- transform(curv, freq)
  as.numeric(res$y[[1]])
}

pattern_rhs_baseline <- function(data, rhs) {
  if (is.null(rhs) || length(rhs) == 0) {
    return(0)
  }
  trans <- pattern_prepare_transactions(data)
  freq <- arules::itemFrequency(trans)
  hits <- freq[names(freq) %in% rhs]
  if (length(hits) == 0) {
    return(0)
  }
  max(hits, na.rm = TRUE)
}
