#'@title Random or SMOTE-based class oversampling
#'@description Balance class distributions by randomly replicating minority examples or by generating synthetic samples with a local SMOTE implementation.
#'@param attribute target class attribute name
#'@param method oversampling strategy: `"smote"` or `"random"`
#'@param k number of nearest neighbors used by the SMOTE strategy
#'@param seed optional random seed for reproducibility
#'@return returns an object of class `bal_oversampling`
#'@references
#' Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique.
#'@examples
#'data(iris)
#'iris_imb <- iris[c(1:50, 51:71, 101:111), ]
#'bal <- bal_oversampling("Species", method = "smote", seed = 123)
#'iris_bal <- transform(bal, iris_imb)
#'table(iris_bal$Species)
#'@export
bal_oversampling <- function(attribute, method = c("smote", "random"), k = 5, seed = NULL) {
  obj <- dal_transform()
  obj$attribute <- attribute
  obj$method <- match.arg(method)
  obj$k <- as.integer(k)
  obj$seed <- seed
  class(obj) <- append("bal_oversampling", class(obj))
  return(obj)
}

bal_numeric_matrix <- function(data, features) {
  x <- data[, features, drop = FALSE]
  x <- as.data.frame(lapply(x, function(col) {
    if (is.numeric(col)) {
      return(col)
    }
    as.numeric(as.factor(col))
  }))
  as.matrix(x)
}

bal_random_oversample <- function(class_data, target_n) {
  if (nrow(class_data) == 0 || target_n <= nrow(class_data)) {
    return(class_data[0, , drop = FALSE])
  }

  idx <- sample(seq_len(nrow(class_data)), size = target_n - nrow(class_data), replace = TRUE)
  class_data[idx, , drop = FALSE]
}

bal_smote_oversample <- function(class_data, target_n, attribute, k) {
  if (nrow(class_data) == 0 || target_n <= nrow(class_data)) {
    return(class_data[0, , drop = FALSE])
  }

  features <- setdiff(names(class_data), attribute)
  if (length(features) == 0) {
    return(bal_random_oversample(class_data, target_n))
  }

  x <- bal_numeric_matrix(class_data, features)
  n <- nrow(x)
  if (n < 2) {
    return(bal_random_oversample(class_data, target_n))
  }

  k <- max(1L, min(as.integer(k), n - 1L))
  synth_n <- target_n - n
  synth_rows <- vector("list", synth_n)

  for (i in seq_len(synth_n)) {
    base_idx <- sample(seq_len(n), size = 1)
    dist <- rowSums((x - matrix(x[base_idx, ], nrow = n, ncol = ncol(x), byrow = TRUE))^2)
    neighbors <- setdiff(order(dist), base_idx)
    if (length(neighbors) == 0) {
      synth_rows[[i]] <- class_data[base_idx, , drop = FALSE]
      next
    }

    nn_idx <- sample(head(neighbors, k), size = 1)
    gap <- stats::runif(1)
    synthetic <- class_data[base_idx, , drop = FALSE]

    for (feature in features) {
      base_val <- class_data[[feature]][base_idx]
      nn_val <- class_data[[feature]][nn_idx]
      if (is.numeric(class_data[[feature]])) {
        synthetic[[feature]] <- base_val + gap * (nn_val - base_val)
      } else {
        synthetic[[feature]] <- sample(c(base_val, nn_val), size = 1)
      }
    }

    synthetic[[attribute]] <- class_data[[attribute]][base_idx]
    synth_rows[[i]] <- synthetic
  }

  do.call(rbind, synth_rows)
}

#'@exportS3Method transform bal_oversampling
transform.bal_oversampling <- function(obj, data, ...) {
  data <- adjust_data.frame(data)
  attribute <- obj$attribute
  if (is.null(attribute) || !attribute %in% names(data)) {
    stop("bal_oversampling: attribute not found in data.")
  }
  if (!is.null(obj$seed)) {
    set.seed(obj$seed)
  }

  counts <- table(data[[attribute]])
  target_n <- max(counts)
  classes <- names(counts)
  parts <- vector("list", length(classes))

  for (i in seq_along(classes)) {
    class_data <- data[data[[attribute]] == classes[i], , drop = FALSE]
    synthetic <- if (obj$method == "random") {
      bal_random_oversample(class_data, target_n)
    } else {
      bal_smote_oversample(class_data, target_n, attribute, obj$k)
    }
    parts[[i]] <- rbind(class_data, synthetic)
  }

  result <- do.call(rbind, parts)
  rownames(result) <- NULL
  return(result)
}
