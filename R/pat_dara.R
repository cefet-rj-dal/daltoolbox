#' @title Tidy association rules
#' @description Convert association rules into a tabular representation suitable for downstream analysis.
#' @param rules An `arules` rules object.
#' @param transactions Optional transaction data used to compute additional interest measures.
#' @param item_columns Optional columns to recreate from left-hand-side items.
#' @return A data frame with rule labels, quality measures, and optional item columns.
#' @examples
#' if (requireNamespace("arules", quietly = TRUE)) {
#'   data("AdultUCI", package = "arules")
#'   data <- as.data.frame(AdultUCI)
#'   data <- data[, c("workclass", "education", "marital-status", "occupation", "income")]
#'   pm <- fit(pat_apriori(target = "rules", supp = 0.05, conf = 0.6, rhs = "income=small"), data)
#'   rules <- pat_rules_tidy(discover(pm, data), transactions = data)
#'   head(rules)
#' }
#' @export
pat_rules_tidy <- function(rules, transactions = NULL, item_columns = NULL) {
  if (!requireNamespace("arules", quietly = TRUE)) {
    stop("pat_rules_tidy requires the 'arules' package.", call. = FALSE)
  }
  if (!inherits(rules, "rules")) {
    stop("pat_rules_tidy expects an 'arules' rules object.", call. = FALSE)
  }
  if (length(rules) == 0) {
    out <- data.frame(lhs = character(), rhs = character(), stringsAsFactors = FALSE)
    if (!is.null(item_columns)) {
      for (name in item_columns) {
        out[[name]] <- character()
      }
    }
    return(out)
  }

  quality <- as.data.frame(arules::quality(rules))
  lhs <- pat_dara_clean_rule_side(arules::labels(arules::lhs(rules)))
  rhs <- pat_dara_clean_rule_side(arules::labels(arules::rhs(rules)))

  out <- data.frame(
    lhs = lhs,
    rhs = rhs,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  for (name in colnames(quality)) {
    value <- quality[[name]]
    if (length(value) == nrow(out)) {
      out[[name]] <- value
    }
  }

  if (!is.null(transactions) && length(rules) > 0) {
    trans <- pattern_prepare_transactions(transactions)
    measures <- arules::interestMeasure(
      rules,
      measure = c("lift", "count", "kulczynski", "imbalance"),
      transactions = trans
    )
    measures <- as.data.frame(measures)
    if (nrow(measures) == nrow(out)) {
      if ("lift" %in% colnames(measures)) out$lift <- measures$lift
      if ("count" %in% colnames(measures)) out$count <- measures$count
      if ("kulczynski" %in% colnames(measures)) out$kulc <- measures$kulczynski
      if ("imbalance" %in% colnames(measures)) out$ir <- measures$imbalance
    }
  }

  if (!is.null(item_columns)) {
    for (name in item_columns) {
      out[[name]] <- NA_character_
    }
    parsed <- lapply(out$lhs, pat_dara_parse_items)
    for (i in seq_along(parsed)) {
      values <- parsed[[i]]
      common <- intersect(names(values), item_columns)
      for (name in common) {
        value <- unname(values[name])
        if (length(value) == 1 && !is.na(value)) {
          out[i, name] <- value
        }
      }
    }
  }

  out
}

#' @title Divergent Association Ranking Analysis
#' @description Rank attribute values whose ordering diverges between the dataset and discovered rules.
#' @details
#' DARA compares two value-frequency orderings for each explanatory attribute:
#' the ordering in a reference dataset, optionally restricted to a right-hand-side
#' value, and the ordering in a table of association rules. The right-hand-side
#' attribute is required so it can be excluded from explanatory divergence.
#' The returned counter `c` increases for values that are more prominent in rules
#' than in the reference data and decreases for values that are less prominent in
#' rules.
#' @param data Reference data frame.
#' @param rules Tidy rule data frame produced by `pat_rules_tidy()`.
#' @param rhs_attr Right-hand-side attribute used by the mined rules.
#' @param rhs_value Optional right-hand-side value used to restrict `data`.
#' @param attributes Optional attributes to compare. Defaults to common columns in `data` and `rules`.
#' @param rule_weight Optional numeric column in `rules` used as rule frequency instead of one count per rule.
#' @return A list of per-attribute data frames with columns `value`, `dataset`, `rules`, and `c`.
#' @references
#' Baroni, L., Salles, R., Salles, S., Guedes, G., Porto, F., Bezerra, E., Barcellos, C.,
#' Pedroso, M., & Ogasawara, E. (2020). An analysis of malaria in the Brazilian Legal
#' Amazon using divergent association rules. Journal of Biomedical Informatics, 108, 103512.
#'
#' The canonical implementation is maintained in the Python `pydaltoolbox` port; this is a
#' backport of it.
#' @examples
#' if (requireNamespace("arules", quietly = TRUE)) {
#'   data("AdultUCI", package = "arules")
#'   data <- as.data.frame(AdultUCI)
#'   data <- data[, c("workclass", "education", "marital-status", "occupation", "income")]
#'   pm <- fit(pat_apriori(target = "rules", supp = 0.05, conf = 0.6, rhs = "income=small"), data)
#'   rules <- pat_rules_tidy(discover(pm, data), transactions = data, item_columns = colnames(data))
#'   ranking <- pat_dara(data, rules, rhs_attr = "income", rhs_value = "small")
#'   ranking[["marital-status"]]
#' }
#' @export
pat_dara <- function(data,
                     rules,
                     rhs_attr,
                     rhs_value = NULL,
                     attributes = NULL,
                     rule_weight = NULL) {
  data <- adjust_data.frame(data)
  rules <- adjust_data.frame(rules)
  rules <- pat_dara_expand_lhs_columns(rules, data)

  if (missing(rhs_attr) || is.null(rhs_attr) || length(rhs_attr) != 1 || is.na(rhs_attr)) {
    stop("pat_dara requires the right-hand-side attribute in 'rhs_attr'.", call. = FALSE)
  }
  if (!rhs_attr %in% colnames(data)) {
    stop("pat_dara: 'rhs_attr' was not found in data.", call. = FALSE)
  }
  if (!is.null(rhs_value)) {
    data <- data[data[[rhs_attr]] == rhs_value, , drop = FALSE]
  }

  if (is.null(attributes)) {
    attributes <- intersect(colnames(data), colnames(rules))
    attributes <- setdiff(attributes, rhs_attr)
  }
  attributes <- attributes[attributes %in% colnames(data) & attributes %in% colnames(rules)]

  results <- list()
  for (attribute in attributes) {
    data_freq <- as.data.frame(table(data[[attribute]], useNA = "no"), stringsAsFactors = FALSE)
    colnames(data_freq) <- c("value", "dataset")
    data_freq$value <- as.character(data_freq$value)

    rule_freq <- pat_dara_rule_frequency(rules, attribute, rule_weight)
    if (nrow(rule_freq) == 0) {
      next
    }

    merged <- merge(data_freq, rule_freq, by = "value", all.x = TRUE)
    merged$rules[is.na(merged$rules)] <- 0
    merged$dataset <- as.numeric(merged$dataset)
    merged$rules <- as.numeric(merged$rules)
    merged$c <- pat_dara_counter(merged$dataset, merged$rules)
    merged <- merged[order(merged$c, merged$value), , drop = FALSE]
    rownames(merged) <- NULL
    results[[attribute]] <- merged
  }

  class(results) <- c("pat_dara", class(results))
  results
}

pat_dara_clean_rule_side <- function(labels) {
  labels <- gsub("^\\{", "", labels)
  labels <- gsub("\\}$", "", labels)
  labels
}

pat_dara_parse_items <- function(label) {
  if (is.na(label) || label == "") {
    return(character())
  }
  items <- trimws(strsplit(label, ",", fixed = TRUE)[[1]])
  values <- character()
  for (item in items) {
    parts <- strsplit(item, "=", fixed = TRUE)[[1]]
    if (length(parts) >= 2) {
      key <- trimws(parts[[1]])
      value <- trimws(paste(parts[-1], collapse = "="))
      values[key] <- value
    }
  }
  values
}

pat_dara_expand_lhs_columns <- function(rules, data) {
  if (!is.data.frame(rules) || !"lhs" %in% colnames(rules) || !(is.data.frame(data) || is.matrix(data))) {
    return(rules)
  }
  item_columns <- setdiff(colnames(data), colnames(rules))
  if (length(item_columns) == 0) {
    return(rules)
  }
  for (name in item_columns) {
    rules[[name]] <- NA_character_
  }
  parsed <- lapply(rules$lhs, pat_dara_parse_items)
  for (i in seq_along(parsed)) {
    values <- parsed[[i]]
    common <- intersect(names(values), item_columns)
    for (name in common) {
      value <- unname(values[name])
      if (length(value) == 1 && !is.na(value)) {
        rules[i, name] <- value
      }
    }
  }
  rules
}

pat_dara_rule_frequency <- function(rules, attribute, rule_weight = NULL) {
  values <- rules[[attribute]]
  keep <- !is.na(values) & values != ""
  if (!any(keep)) {
    return(data.frame(value = character(), rules = numeric()))
  }

  if (!is.null(rule_weight)) {
    if (!rule_weight %in% colnames(rules)) {
      stop("pat_dara: 'rule_weight' was not found in rules.", call. = FALSE)
    }
    weights <- as.numeric(rules[[rule_weight]])
    weights[is.na(weights)] <- 0
    out <- stats::aggregate(weights[keep], by = list(value = as.character(values[keep])), FUN = sum)
    colnames(out) <- c("value", "rules")
    return(out)
  }

  out <- as.data.frame(table(values[keep]), stringsAsFactors = FALSE)
  colnames(out) <- c("value", "rules")
  out$value <- as.character(out$value)
  out
}

pat_dara_counter <- function(dataset, rules) {
  counter <- rep(0, length(dataset))
  n <- length(counter)
  if (n <= 1) {
    return(counter)
  }
  for (j in seq_len(n - 1)) {
    for (k in (j + 1):n) {
      if ((dataset[[j]] > dataset[[k]]) && (rules[[j]] < rules[[k]])) {
        counter[[j]] <- counter[[j]] - 1
        counter[[k]] <- counter[[k]] + 1
      } else if ((dataset[[j]] < dataset[[k]]) && (rules[[j]] > rules[[k]])) {
        counter[[j]] <- counter[[j]] + 1
        counter[[k]] <- counter[[k]] - 1
      }
    }
  }
  counter
}
