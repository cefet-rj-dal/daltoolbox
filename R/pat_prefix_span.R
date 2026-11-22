#'@title PrefixSpan sequences
#'@description Frequent sequential pattern mining with PrefixSpan.
#'@details
#' PrefixSpan (Prefix-projected Sequential pattern mining) discovers frequent sequential
#' patterns -- ordered lists of itemsets -- by growing patterns one prefix at a time and
#' recursively mining the projected database of suffixes. A pattern grows by an i-extension
#' (adding an item to its last itemset) or an s-extension (appending a new itemset); the
#' support of a pattern is the fraction of input sequences in which it occurs in order.
#' `maxlen` bounds the number of itemsets, `maxsize` the items per itemset, and
#' `mingap`/`maxgap` constrain the time difference between consecutive matched events.
#'
#' Sequences are given as a data frame with `sequenceID`, `eventID` and `item` columns, or
#' as a list of sequences (each a list of itemsets, or of `list(time, items)` events).
#'@param support minimum support threshold. If `0`, estimated during `fit()` using `support_strategy`.
#'@param support_strategy support threshold strategy created with `pat_support_threshold()`
#'@param minlen minimum number of events per pattern
#'@param maxlen maximum number of events per pattern (`NULL` = unbounded)
#'@param maxsize maximum number of items per event (`NULL` = unbounded)
#'@param mingap minimum gap between successive events (`NULL` = unbounded)
#'@param maxgap maximum gap between successive events (`NULL` = unbounded)
#'@param quality_filter optional quality filter created with `patutils()`
#'@return returns a `pat_prefix_span` object
#'@references
#' Pei, J., Han, J., Mortazavi-Asl, B., Wang, J., Pinto, H., Chen, Q., Dayal, U., & Hsu, M.
#' (2001). PrefixSpan: Mining Sequential Patterns Efficiently by Prefix-Projected Pattern
#' Growth. ICDE, 215-224.
#'
#' The canonical implementation is maintained in the Python `pydaltoolbox` port; this is a
#' backport of it.
#'@examples
#' seqs <- data.frame(
#'   sequenceID = c(1,1,1,2,2,3),
#'   eventID    = c(10,20,20,10,20,10),
#'   item       = c("A","B","C","A","C","A")
#' )
#' pm <- pat_prefix_span(support = 0.5)
#' pm <- fit(pm, seqs)
#' seqs_found <- discover(pm, seqs)
#' evaluate(pm, seqs_found)$metrics
#'@export
pat_prefix_span <- function(support = 0,
                            support_strategy = pat_support_threshold("min_count", min_count = 2),
                            minlen = 1,
                            maxlen = NULL,
                            maxsize = NULL,
                            mingap = NULL,
                            maxgap = NULL,
                            quality_filter = NULL) {
  obj <- pattern_miner()
  utils <- obj$pat_utils
  obj$support <- support
  obj$support_strategy <- support_strategy
  obj$minlen <- minlen
  obj$maxlen <- maxlen
  obj$maxsize <- maxsize
  obj$mingap <- mingap
  obj$maxgap <- maxgap
  obj$quality_filter <- quality_filter
  obj$pattern_kind <- "sequences"
  obj$eval_metrics <- list(
    utils$metric_pattern_count,
    utils$metric_mean_support,
    utils$metric_mean_length,
    utils$metric_retained_ratio
  )
  class(obj) <- append("pat_prefix_span", class(obj))
  return(obj)
}

#'@exportS3Method fit pat_prefix_span
fit.pat_prefix_span <- function(obj, data, ...) {
  sequences <- pat_prefix_span_normalize(data)
  obj$sequences <- sequences
  items <- unlist(lapply(sequences, function(seq) unlist(lapply(seq, function(e) e$items))))
  obj$schema <- sort(unique(items))
  obj$data_class <- class(data)[1]
  obj$fitted <- TRUE
  if (!is.null(obj$support) && obj$support > 0) {
    obj$support_resolved <- obj$support
  } else {
    onehot <- pat_prefix_span_onehot(sequences)
    obj$support_resolved <- pat_resolve_support(0, obj$support_strategy, onehot)
  }
  obj
}

#'@exportS3Method discover pat_prefix_span
discover.pat_prefix_span <- function(obj, data, ...) {
  pattern_miner_require_fitted(obj)
  sequences <- if (missing(data) || is.null(data)) obj$sequences else pat_prefix_span_normalize(data)
  n <- length(sequences)
  min_count <- max(1, ceiling(obj$support_resolved * n - 1e-9))

  mined <- pat_prefix_span_mine(
    sequences, min_count,
    maxlen = if (is.null(obj$maxlen)) .Machine$integer.max else obj$maxlen,
    maxsize = if (is.null(obj$maxsize)) .Machine$integer.max else obj$maxsize,
    mingap = obj$mingap,
    maxgap = obj$maxgap
  )

  if (length(mined) == 0) {
    out <- data.frame(sequence = character(), support = numeric(), length = integer(),
                      stringsAsFactors = FALSE)
    attr(out, "filtered_from") <- 0
    return(out)
  }
  out <- data.frame(
    sequence = vapply(mined, function(m) pat_prefix_span_label(m$pattern), character(1)),
    support = vapply(mined, function(m) m$count / n, numeric(1)),
    length = vapply(mined, function(m) length(m$pattern), integer(1)),
    stringsAsFactors = FALSE
  )
  out <- out[out$length >= obj$minlen, , drop = FALSE]
  total <- nrow(out)

  filter <- obj$quality_filter
  if (!is.null(filter)) {
    if (!is.null(filter$min$support)) out <- out[out$support >= filter$min$support, , drop = FALSE]
    if (!is.null(filter$max$support)) out <- out[out$support <= filter$max$support, , drop = FALSE]
  }
  out <- out[order(-out$support, out$sequence), , drop = FALSE]
  rownames(out) <- NULL
  attr(out, "filtered_from") <- total
  out
}

#'@exportS3Method evaluate pat_prefix_span
evaluate.pat_prefix_span <- function(obj, patterns, ...) {
  total <- attr(patterns, "filtered_from", exact = TRUE)
  if (is.null(total)) total <- nrow(patterns)
  np <- nrow(patterns)
  metrics <- data.frame(
    metric = c("pattern_count", "mean_support", "mean_length", "retained_ratio"),
    value = c(
      np,
      if (np > 0) mean(patterns$support) else NA_real_,
      if (np > 0) mean(patterns$length) else NA_real_,
      if (!is.null(total) && total > 0) np / total else NA_real_
    ),
    type = c("intrinsic", "intrinsic", "intrinsic", "filter"),
    stringsAsFactors = FALSE
  )
  list(patterns = patterns, metrics = metrics)
}

# -- helpers ------------------------------------------------------------------

pat_prefix_span_normalize <- function(data) {
  if (is.data.frame(data)) {
    cn <- tolower(colnames(data))
    sid <- colnames(data)[match("sequenceid", cn)]
    eid <- colnames(data)[match("eventid", cn)]
    item <- colnames(data)[match("item", cn)]
    if (any(is.na(c(sid, eid, item)))) {
      stop("pat_prefix_span: data frame input needs 'sequenceID', 'eventID' and 'item' columns.",
           call. = FALSE)
    }
    sequences <- list()
    for (s in sort(unique(data[[sid]]))) {
      sub <- data[data[[sid]] == s, , drop = FALSE]
      events <- list()
      for (ev in sort(unique(sub[[eid]]))) {
        items <- sort(unique(as.character(sub[[item]][sub[[eid]] == ev])))
        events[[length(events) + 1]] <- list(time = ev, items = items)
      }
      sequences[[length(sequences) + 1]] <- events
    }
    return(sequences)
  }
  if (is.list(data)) {
    sequences <- list()
    for (seq in data) {
      events <- list()
      t <- 0
      for (event in seq) {
        t <- t + 1
        if (is.list(event) && !is.null(event$items)) {
          events[[length(events) + 1]] <- list(
            time = if (!is.null(event$time)) event$time else t,
            items = sort(unique(as.character(event$items)))
          )
        } else {
          events[[length(events) + 1]] <- list(time = t, items = sort(unique(as.character(event))))
        }
      }
      events <- events[order(vapply(events, function(e) e$time, numeric(1)))]
      sequences[[length(sequences) + 1]] <- events
    }
    return(sequences)
  }
  stop("pat_prefix_span: unsupported data type for sequence mining.", call. = FALSE)
}

pat_prefix_span_onehot <- function(sequences) {
  items <- sort(unique(unlist(lapply(sequences, function(seq) unlist(lapply(seq, function(e) e$items))))))
  rows <- lapply(sequences, function(seq) {
    present <- unique(unlist(lapply(seq, function(e) e$items)))
    stats::setNames(items %in% present, items)
  })
  as.data.frame(do.call(rbind, rows))
}

pat_prefix_span_label <- function(pattern) {
  paste0("<", paste0("{", vapply(pattern, function(e) paste(e, collapse = ","), character(1)), "}",
                     collapse = ","), ">")
}

pat_prefix_span_mine <- function(sequences, min_count, maxlen, maxsize, mingap, maxgap) {
  results <- list()
  recurse <- function(prefix, proj, is_empty) {
    n_elem <- length(prefix)
    last_item <- if (!is_empty) prefix[[n_elem]][length(prefix[[n_elem]])] else NULL
    last_size <- if (!is_empty) length(prefix[[n_elem]]) else 0L

    i_cand <- list()
    s_cand <- list()
    for (si in names(proj)) {
      eset <- proj[[si]]
      seq <- sequences[[as.integer(si)]]
      if (!is_empty && last_size < maxsize) {
        for (e in eset) {
          for (x in seq[[e]]$items) {
            if (x > last_item) i_cand[[x]][[si]] <- unique(c(i_cand[[x]][[si]], e))
          }
        }
      }
      if (n_elem < maxlen) {
        for (e_prev in eset) {
          t_prev <- if (e_prev < 1) NA_real_ else seq[[e_prev]]$time
          e2 <- e_prev + 1
          while (e2 <= length(seq)) {
            if (!is.na(t_prev)) {
              gap <- seq[[e2]]$time - t_prev
              if (!is.null(maxgap) && gap > maxgap) break
              if (!is.null(mingap) && gap < mingap) { e2 <- e2 + 1; next }
            }
            for (x in seq[[e2]]$items) s_cand[[x]][[si]] <- unique(c(s_cand[[x]][[si]], e2))
            e2 <- e2 + 1
          }
        }
      }
    }

    for (x in sort(names(i_cand))) {
      seqs <- i_cand[[x]]
      if (length(seqs) >= min_count) {
        newprefix <- prefix
        newprefix[[n_elem]] <- c(newprefix[[n_elem]], x)
        results[[length(results) + 1]] <<- list(pattern = newprefix, count = length(seqs))
        recurse(newprefix, seqs, FALSE)
      }
    }
    for (x in sort(names(s_cand))) {
      seqs <- s_cand[[x]]
      if (length(seqs) >= min_count) {
        newprefix <- c(prefix, list(x))
        results[[length(results) + 1]] <<- list(pattern = newprefix, count = length(seqs))
        recurse(newprefix, seqs, FALSE)
      }
    }
  }
  proj0 <- stats::setNames(as.list(rep(0L, length(sequences))), as.character(seq_along(sequences)))
  recurse(list(), proj0, TRUE)
  results
}
