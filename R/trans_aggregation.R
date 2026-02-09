#'@title Aggregation by groups
#'@description Aggregate data by a grouping attribute using named expressions.
#'@param group grouping column name (string)
#'@param ... named expressions evaluated per group
#'@return returns an object of class `aggregation`
#'@examples
#'data(iris)
#'agg <- aggregation(
#'  "Species",
#'  mean_sepal = mean(Sepal.Length),
#'  sd_sepal = sd(Sepal.Length),
#'  n = n()
#')
#'iris_agg <- transform(agg, iris)
#'iris_agg
#'@export
aggregation <- function(group, ...) {
  obj <- dal_transform()
  obj$group <- group
  obj$exprs <- as.list(substitute(list(...)))[-1]
  class(obj) <- append("aggregation", class(obj))
  return(obj)
}

#'@exportS3Method transform aggregation
transform.aggregation <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  group <- obj$group
  if (is.null(group) || !group %in% names(data)) {
    stop("aggregation: 'group' must be a valid column name in data.")
  }
  exprs <- obj$exprs
  if (length(exprs) == 0) {
    stop("aggregation: no aggregation expressions provided.")
  }
  if (is.null(names(exprs)) || any(names(exprs) == "")) {
    stop("aggregation: all aggregation expressions must be named.")
  }

  groups <- split(seq_len(nrow(data)), data[[group]])
  group_values <- names(groups)
  if (is.factor(data[[group]])) {
    group_values <- factor(group_values, levels = levels(data[[group]]))
  }

  rows <- vector("list", length(groups))
  for (i in seq_along(groups)) {
    idx <- groups[[i]]
    env_data <- data[idx, , drop=FALSE]
    env <- list2env(as.list(env_data), parent = parent.frame())
    env$n <- function() nrow(env_data)
    row <- list()
    row[[group]] <- group_values[i]
    for (nm in names(exprs)) {
      value <- eval(exprs[[nm]], envir = env, enclos = parent.frame())
      if (is.function(value)) {
        value <- value()
      }
      row[[nm]] <- value
    }
    rows[[i]] <- as.data.frame(row, stringsAsFactors = FALSE)
  }

  result <- do.call(rbind, rows)
  rownames(result) <- NULL
  return(result)
}
