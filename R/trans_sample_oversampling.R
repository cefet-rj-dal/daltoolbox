#'@title SMOTE-family class oversampling
#'@description Balance class distributions by generating synthetic minority examples with
#' the SMOTE family (classic SMOTE, Borderline-SMOTE, ADASYN) or by random replication.
#'@details All synthetic variants interpolate a base minority instance toward one of its
#' `k` nearest minority neighbors; nominal features are set by majority vote among those
#' neighbors (SMOTE-NC) instead of a coin flip. `"borderline"` synthesizes only from
#' *danger* instances (those whose `m` nearest neighbors are mostly of other classes),
#' concentrating new points near the decision boundary; `"adasyn"` allocates more synthetic
#' samples to harder-to-learn minority instances.
#'@param attribute target class attribute name
#'@param method oversampling strategy: `"smote"`, `"borderline"`, `"adasyn"` or `"random"`
#'@param k number of nearest neighbors used to interpolate synthetic samples
#'@param m number of neighbors used to assess the danger/difficulty of a minority instance (borderline/adasyn)
#'@param seed optional seed for reproducibility
#'@return returns an object of class `bal_oversampling`
#'@references
#' Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16:321-357.
#' Han, H., Wang, W.-Y., & Mao, B.-H. (2005). Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning. In ICIC 2005, 878-887.
#' He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning. In IJCNN 2008, 1322-1328.
#'@examples
#'data(iris)
#'iris_imb <- iris[c(1:50, 51:71, 101:111), ]
#'bal <- bal_oversampling("Species", method = "smote")
#'iris_bal <- transform(bal, iris_imb)
#'table(iris_bal$Species)
#'@importFrom utils head
#'@export
bal_oversampling <- function(attribute, method = c("smote", "borderline", "adasyn", "random"), k = 5, m = 10, seed = NULL) {
  obj <- dal_transform()
  obj$attribute <- attribute
  obj$method <- match.arg(method)
  obj$k <- as.integer(k)
  obj$m <- as.integer(m)
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

# Fraction of each minority instance's m nearest (whole-data) neighbors in other classes.
bal_other_ratio <- function(data, cls, attribute, features, m) {
  all_x <- bal_numeric_matrix(data, features)
  is_other <- as.character(data[[attribute]]) != as.character(cls)
  mem <- as.character(data[[attribute]]) == as.character(cls)
  mem_x <- all_x[mem, , drop = FALSE]
  nmem <- nrow(mem_x)
  mm <- max(1L, as.integer(m))
  ratios <- numeric(nmem)
  for (i in seq_len(nmem)) {
    d <- rowSums((all_x - matrix(mem_x[i, ], nrow = nrow(all_x), ncol = ncol(all_x), byrow = TRUE))^2)
    nn <- order(d)[-1][seq_len(min(mm, length(d) - 1L))]  # skip self at position 1
    ratios[i] <- mean(is_other[nn])
  }
  ratios
}

# Base minority positions to synthesize from, per method.
bal_bases <- function(data, cls, class_data, need, attribute, features, method, k, m) {
  n <- nrow(class_data)
  if (method == "smote") {
    return(sample.int(n, size = need, replace = TRUE))
  }
  ratios <- bal_other_ratio(data, cls, attribute, features, m)
  if (method == "borderline") {
    danger <- which(ratios >= 0.5 & ratios < 1.0)
    pool <- if (length(danger) > 0) danger else seq_len(n)
    return(pool[sample.int(length(pool), size = need, replace = TRUE)])
  }
  # adasyn: sample bases proportional to their difficulty ratio
  w <- if (sum(ratios) > 0) ratios else rep(1, n)
  sample.int(n, size = need, replace = TRUE, prob = w / sum(w))
}

# One synthetic row: interpolate base toward a random nearest minority neighbor;
# nominal features by majority vote among the base's k nearest minority neighbors (SMOTE-NC).
bal_make_one <- function(class_data, x, base_pos, features, attribute, k) {
  n <- nrow(x)
  dist <- rowSums((x - matrix(x[base_pos, ], nrow = n, ncol = ncol(x), byrow = TRUE))^2)
  neighbors <- setdiff(order(dist), base_pos)
  if (length(neighbors) == 0) {
    return(class_data[base_pos, , drop = FALSE])
  }
  kk <- max(1L, min(as.integer(k), n - 1L))
  neighbors <- head(neighbors, kk)
  nn_idx <- neighbors[sample.int(length(neighbors), 1)]
  gap <- stats::runif(1)
  synthetic <- class_data[base_pos, , drop = FALSE]
  for (feature in features) {
    if (is.numeric(class_data[[feature]])) {
      base_val <- class_data[[feature]][base_pos]
      nn_val <- class_data[[feature]][nn_idx]
      synthetic[[feature]] <- base_val + gap * (nn_val - base_val)
    } else {
      vals <- class_data[[feature]][neighbors]
      maj <- names(which.max(table(vals)))
      if (is.factor(class_data[[feature]])) {
        synthetic[[feature]] <- factor(maj, levels = levels(class_data[[feature]]))
      } else {
        synthetic[[feature]] <- maj
      }
    }
  }
  synthetic[[attribute]] <- class_data[[attribute]][base_pos]
  synthetic
}

bal_synth_oversample <- function(data, cls, class_data, target_n, attribute, method, k, m) {
  n <- nrow(class_data)
  if (n == 0 || target_n <= n) {
    return(class_data[0, , drop = FALSE])
  }
  features <- setdiff(names(class_data), attribute)
  if (length(features) == 0 || n < 2) {
    return(bal_random_oversample(class_data, target_n))
  }
  x <- bal_numeric_matrix(class_data, features)
  need <- target_n - n
  bases <- bal_bases(data, cls, class_data, need, attribute, features, method, k, m)
  synth_rows <- vector("list", need)
  for (i in seq_len(need)) {
    synth_rows[[i]] <- bal_make_one(class_data, x, bases[i], features, attribute, k)
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

  if (!is.null(obj$seed)) set.seed(obj$seed)
  counts <- table(data[[attribute]])
  target_n <- max(counts)
  classes <- names(counts)
  parts <- vector("list", length(classes))

  for (i in seq_along(classes)) {
    class_data <- data[data[[attribute]] == classes[i], , drop = FALSE]
    synthetic <- if (obj$method == "random") {
      bal_random_oversample(class_data, target_n)
    } else {
      bal_synth_oversample(data, classes[i], class_data, target_n, attribute, obj$method, obj$k, obj$m)
    }
    parts[[i]] <- rbind(class_data, synthetic)
  }

  result <- do.call(rbind, parts)
  rownames(result) <- NULL
  return(result)
}
