#'@title Tree-based imputation
#'@description Impute missing values in mixed datasets with iterative decision trees.
#'@details The method starts with a simple imputation to obtain a complete working table.
#' Then, for each target column with missing values, it trains a decision tree using the
#' remaining supported columns as predictors. Columns are updated in ascending order of
#' missingness, from the least missing to the most missing, and the process is repeated for
#' a fixed number of iterations.
#'
#' This design follows the general variable-by-variable iterative imputation strategy from
#' the literature and adopts the ordering by missingness used in `missForest`, while keeping
#' the implementation lightweight through single-tree models.
#'@param method initial imputation method for numeric columns: "median" or "mean"
#'@param cols optional vector of target column names to impute (default: all supported columns)
#'@param maxit number of imputation iterations during fitting
#'@return returns an object of class `imputation_tree`
#'@references
#' Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984).
#' Classification and Regression Trees. Wadsworth.
#'
#' Stekhoven, D. J., Buhlmann, P. (2012). MissForest: non-parametric missing value
#' imputation for mixed-type data. Bioinformatics, 28(1), 112-118.
#'
#' van Buuren, S., Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by
#' Chained Equations in R. Journal of Statistical Software, 45(3), 1-67.
#'@examples
#'data(iris)
#'iris_na <- iris
#'iris_na$Sepal.Length[c(2, 10, 25)] <- NA
#'iris_na$Petal.Width[c(5, 11)] <- NA
#'iris_na$Species[c(3, 15)] <- NA
#'
#'imp <- imputation_tree(maxit = 3)
#'imp <- fit(imp, iris_na)
#'iris_imp <- transform(imp, iris_na)
#'summary(iris_imp$Sepal.Length)
#'table(iris_imp$Species, useNA = "ifany")
#'@export
imputation_tree <- function(method = c("median", "mean"), cols = NULL, maxit = 5) {
  if (!is.numeric(maxit) || length(maxit) != 1 || is.na(maxit) || maxit < 1) {
    stop("imputation_tree: 'maxit' must be a positive number.")
  }
  obj <- dal_transform()
  obj$method <- match.arg(method)
  obj$cols <- cols
  obj$maxit <- as.integer(maxit)
  class(obj) <- append("imputation_tree", class(obj))
  return(obj)
}

imputation_tree_get_levels <- function(x, kind) {
  if (kind == "numeric") {
    return(NULL)
  }
  if (kind == "factor" || kind == "ordered") {
    return(levels(x))
  }
  if (kind == "logical") {
    return(c("FALSE", "TRUE"))
  }
  unique(as.character(x[!is.na(x)]))
}

imputation_tree_prepare_column <- function(x, kind, levels) {
  if (kind == "numeric") {
    return(as.numeric(x))
  }

  values <- as.character(x)
  prepared <- factor(values, levels = levels)
  invalid <- !is.na(x) & is.na(prepared)
  if (any(invalid)) {
    stop("imputation_tree: transform data contains unseen categorical levels.")
  }
  return(prepared)
}

imputation_tree_restore_column <- function(x, kind, levels) {
  if (kind == "numeric") {
    return(as.numeric(x))
  }

  values <- as.character(x)
  if (kind == "factor") {
    return(factor(values, levels = levels))
  }
  if (kind == "ordered") {
    return(ordered(values, levels = levels))
  }
  if (kind == "logical") {
    return(as.logical(values))
  }
  return(values)
}

imputation_tree_prepare_frame <- function(data, columns, kinds, levels_map) {
  frame <- data[, columns, drop = FALSE]
  for (col in columns) {
    frame[[col]] <- imputation_tree_prepare_column(frame[[col]], kinds[[col]], levels_map[[col]])
  }
  return(frame)
}

imputation_tree_has_change <- function(old, new, kind) {
  if (kind == "numeric") {
    return(any(abs(old - new) > sqrt(.Machine$double.eps), na.rm = TRUE))
  }
  return(any(as.character(old) != as.character(new), na.rm = TRUE))
}

#'@exportS3Method fit imputation_tree
fit.imputation_tree <- function(obj, data, ...) {
  data <- adjust_data.frame(data)

  supported <- sapply(data, function(col) {
    is.numeric(col) || is.ordered(col) || is.factor(col) || is.character(col) || is.logical(col)
  })
  supported_cols <- names(data)[supported]
  if (length(supported_cols) == 0) {
    stop("imputation_tree: no supported columns available.")
  }

  target_cols <- obj$cols
  if (is.null(target_cols)) {
    target_cols <- supported_cols
  }
  if (!all(target_cols %in% names(data))) {
    stop("imputation_tree: some columns in 'cols' are not in data.")
  }

  supported_kinds <- vapply(data[, supported_cols, drop = FALSE], imputation_simple_kind, character(1))
  levels_map <- lapply(supported_cols, function(col) {
    imputation_tree_get_levels(data[[col]], supported_kinds[[col]])
  })
  names(levels_map) <- supported_cols

  for (col in supported_cols) {
    if (all(is.na(data[[col]]))) {
      stop(paste0("imputation_tree: column '", col, "' contains only missing values."))
    }
  }

  for (col in target_cols) {
    if (all(is.na(data[[col]]))) {
      stop(paste0("imputation_tree: column '", col, "' contains only missing values."))
    }
  }

  missing_counts <- vapply(target_cols, function(col) sum(is.na(data[[col]])), integer(1))
  order <- names(sort(missing_counts[missing_counts > 0], decreasing = FALSE))

  initial_model <- imputation_simple(method = obj$method, cols = supported_cols)
  initial_model <- fit(initial_model, data)
  work <- transform(initial_model, data)

  models <- stats::setNames(vector("list", length(order)), order)
  if (length(order) > 0) {
    for (iter in seq_len(obj$maxit)) {
      changed <- FALSE
      for (col in order) {
        predictors <- setdiff(supported_cols, col)
        if (length(predictors) == 0) {
          models[[col]] <- NULL
          next
        }

        observed <- !is.na(data[[col]])
        train_frame <- work[observed, c(col, predictors), drop = FALSE]
        train_frame <- imputation_tree_prepare_frame(train_frame, c(col, predictors), supported_kinds, levels_map)

        model <- tree::tree(stats::formula(paste(col, "~ .")), train_frame)
        models[[col]] <- list(model = model, predictors = predictors)

        missing_idx <- is.na(data[[col]])
        if (!any(missing_idx)) {
          next
        }

        pred_frame <- work[missing_idx, predictors, drop = FALSE]
        pred_frame <- imputation_tree_prepare_frame(pred_frame, predictors, supported_kinds, levels_map)
        prediction <- if (supported_kinds[[col]] == "numeric") {
          predict(model, pred_frame, type = "vector")
        } else {
          predict(model, pred_frame, type = "class")
        }
        prediction <- imputation_tree_restore_column(prediction, supported_kinds[[col]], levels_map[[col]])

        previous <- work[[col]][missing_idx]
        work[[col]][missing_idx] <- prediction
        if (imputation_tree_has_change(previous, prediction, supported_kinds[[col]])) {
          changed <- TRUE
        }
      }

      if (!changed) {
        break
      }
    }
  }

  obj$supported_cols <- supported_cols
  obj$target_cols <- target_cols
  obj$order <- order
  obj$kinds <- supported_kinds
  obj$levels <- levels_map
  obj$initial_model <- initial_model
  obj$models <- models
  return(obj)
}

#'@exportS3Method transform imputation_tree
transform.imputation_tree <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  data_work <- data

  initial_missing <- lapply(obj$supported_cols, function(col) {
    if (col %in% names(data)) is.na(data[[col]]) else NULL
  })
  names(initial_missing) <- obj$supported_cols

  data_work <- transform(obj$initial_model, data_work)

  for (col in obj$order) {
    if (!col %in% names(data_work)) {
      next
    }

    missing_idx <- initial_missing[[col]]
    if (is.null(missing_idx) || !any(missing_idx)) {
      next
    }

    model_info <- obj$models[[col]]
    if (is.null(model_info)) {
      next
    }

    missing_predictors <- setdiff(model_info$predictors, names(data_work))
    if (length(missing_predictors) > 0) {
      stop(paste0(
        "imputation_tree: missing predictor columns in transform data: ",
        paste(missing_predictors, collapse = ", ")
      ))
    }

    pred_frame <- data_work[missing_idx, model_info$predictors, drop = FALSE]
    pred_frame <- imputation_tree_prepare_frame(pred_frame, model_info$predictors, obj$kinds, obj$levels)
    prediction <- if (obj$kinds[[col]] == "numeric") {
      predict(model_info$model, pred_frame, type = "vector")
    } else {
      predict(model_info$model, pred_frame, type = "class")
    }
    prediction <- imputation_tree_restore_column(prediction, obj$kinds[[col]], obj$levels[[col]])
    data_work[[col]][missing_idx] <- prediction
  }

  non_targets <- setdiff(obj$supported_cols, obj$target_cols)
  for (col in non_targets) {
    if (!col %in% names(data_work)) {
      next
    }
    idx <- initial_missing[[col]]
    if (!is.null(idx) && any(idx)) {
      data_work[[col]][idx] <- NA
    }
  }

  return(data_work)
}
