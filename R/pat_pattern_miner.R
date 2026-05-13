#' @title Pattern Miner
#' @description Base class for frequent pattern and sequence mining.
#' @details
#' Pattern miners follow a lightweight Experiment Line:
#'
#' - `fit()` validates the mining input and stores a schema signature
#' - `discover()` runs the mining algorithm on data compatible with that schema
#' - `evaluate()` summarizes pattern quality and filtering effects
#'
#' Different miners may normalize their inputs differently (for example, item
#' transactions versus sequence transactions), but the base contract remains the
#' same.
#' @return A `pattern_miner` object.
#' @examples
#' miner <- pattern_miner()
#' class(miner)
#' @export
pattern_miner <- function() {
  obj <- dal_learner()
  utils <- patutils()
  obj$schema <- NULL
  obj$data_class <- NULL
  obj$fitted <- FALSE
  obj$pat_utils <- utils
  obj$quality_filter <- NULL
  obj$eval_metrics <- list(
    utils$metric_pattern_count,
    utils$metric_mean_support,
    utils$metric_mean_length,
    utils$metric_retained_ratio
  )
  obj$engine_parameter <- NULL
  obj$engine_appearance <- NULL
  obj$engine_control <- NULL
  obj$pattern_kind <- "patterns"
  class(obj) <- append("pattern_miner", class(obj))
  return(obj)
}

#' @exportS3Method action pattern_miner
action.pattern_miner <- function(obj, ...) {
  thiscall <- match.call(expand.dots = TRUE)
  thiscall[[1]] <- as.name("discover")
  result <- eval.parent(thiscall)
  return(result)
}

#' @title Discover
#' @description Generic for pattern discovery.
#' @param obj A `pattern_miner` object.
#' @param ... Optional arguments passed to the concrete discovery method.
#' @return Discovered patterns in the representation used by the mining engine.
#' @export
discover <- function(obj, ...) {
  UseMethod("discover")
}

#' @exportS3Method discover default
discover.default <- function(obj, ...) {
  return(list())
}

# Infer a lightweight schema signature without keeping the dataset in memory.
pattern_schema <- function(data) {
  if (inherits(data, "transactions")) {
    return(sort(arules::itemLabels(data)))
  }
  if (is.data.frame(data) || is.matrix(data)) {
    return(sort(colnames(data)))
  }
  stop("pattern_miner: unsupported data type for schema inference.")
}

pattern_miner_mark_fitted <- function(obj, data) {
  obj$schema <- pattern_schema(data)
  obj$data_class <- class(data)[1]
  obj$fitted <- TRUE
  obj
}

pattern_miner_require_fitted <- function(obj) {
  if (!isTRUE(obj$fitted)) {
    stop(sprintf("%s must be fitted before discovery.", class(obj)[1]), call. = FALSE)
  }
  invisible(TRUE)
}

pattern_miner_apply_quality_filter <- function(obj, patterns) {
  filter <- obj$quality_filter
  utils <- if (!is.null(obj$pat_utils)) obj$pat_utils else patutils()
  total <- attr(patterns, "filtered_from", exact = TRUE)
  if (is.null(total) || is.na(total)) {
    total <- length(patterns)
  }
  if (is.null(filter)) {
    attr(patterns, "filtered_from") <- total
    return(patterns)
  }

  quality <- utils$pattern_quality(patterns)
  idx <- rep(TRUE, nrow(quality))

  if (!is.null(filter$min)) {
    for (name in names(filter$min)) {
      if (name %in% colnames(quality)) {
        idx <- idx & quality[[name]] >= filter$min[[name]]
      }
    }
  }

  if (!is.null(filter$max)) {
    for (name in names(filter$max)) {
      if (name %in% colnames(quality)) {
        idx <- idx & quality[[name]] <= filter$max[[name]]
      }
    }
  }

  patterns <- patterns[idx]
  attr(patterns, "filtered_from") <- total
  patterns
}

pattern_miner_apply_item_filter <- function(obj, patterns) {
  include <- obj$include
  exclude <- obj$exclude
  utils <- if (!is.null(obj$pat_utils)) obj$pat_utils else patutils()

  if (is.null(include) && is.null(exclude)) {
    attr(patterns, "filtered_from") <- length(patterns)
    return(patterns)
  }

  total <- length(patterns)
  labels <- arules::labels(patterns)
  tokens <- utils$item_tokens(labels)
  idx <- rep(TRUE, length(tokens))

  if (!is.null(include)) {
    idx <- idx & vapply(tokens, function(items) all(items %in% include), logical(1))
  }
  if (!is.null(exclude)) {
    idx <- idx & vapply(tokens, function(items) all(!items %in% exclude), logical(1))
  }

  patterns <- patterns[idx]
  attr(patterns, "filtered_from") <- total
  patterns
}

pattern_prepare_transactions <- function(data) {
  if (inherits(data, "transactions")) {
    return(data)
  }
  if (is.data.frame(data) || is.matrix(data)) {
    return(methods::as(adjust_data.frame(data), "transactions"))
  }
  stop("pattern_miner: unsupported data type for transaction mining.", call. = FALSE)
}

pattern_prepare_sequences <- function(data) {
  if (inherits(data, "transactions")) {
    return(data)
  }
  stop("pattern_miner: sequence mining expects a 'transactions' object produced by arulesSequences::read_baskets().", call. = FALSE)
}

validate_pattern_schema <- function(obj, data) {
  if (is.null(obj$schema)) {
    return(invisible(TRUE))
  }
  current <- pattern_schema(data)
  if (!identical(obj$schema, current)) {
    stop("pattern_miner: discover data schema differs from fit schema.")
  }
  invisible(TRUE)
}

#' @exportS3Method fit pattern_miner
fit.pattern_miner <- function(obj, data, ...) {
  pattern_miner_mark_fitted(obj, data)
}

#' @exportS3Method evaluate pattern_miner
evaluate.pattern_miner <- function(obj, patterns, ...) {
  utils <- if (!is.null(obj$pat_utils)) obj$pat_utils else patutils()
  metrics <- obj$eval_metrics
  if (is.null(metrics)) {
    metrics <- list(utils$metric_pattern_count)
  }

  rows <- NULL
  for (metric_fn in metrics) {
    metric_res <- metric_fn(patterns = patterns, obj = obj)
    rows <- rbind(rows, data.frame(
      metric = metric_res$metric,
      value = metric_res$value,
      type = metric_res$type,
      check.names = FALSE
    ))
  }

  list(
    patterns = patterns,
    metrics = rows
  )
}
