# installation
# install.packages(c("daltoolbox", "smotefamily"))

library(daltoolbox)

smote_custom <- function(attribute) {
  obj <- daltoolbox::dal_transform()
  obj$attribute <- attribute
  class(obj) <- append("smote_custom", class(obj))
  obj
}

transform.smote_custom <- function(obj, data, ...) {
  if (!requireNamespace("smotefamily", quietly = TRUE)) {
    stop("This example requires the 'smotefamily' package.")
  }

  data <- data.frame(data, check.names = FALSE)
  j <- match(obj$attribute, colnames(data))
  x <- sort(table(data[, obj$attribute]))
  result <- data[data[obj$attribute] == names(x)[length(x)], , drop = FALSE]

  for (i in seq_len(length(x) - 1)) {
    small_name <- names(x)[i]
    large_name <- names(x)[length(x)]
    small <- data[, obj$attribute] == small_name
    large <- data[, obj$attribute] == large_name
    data_smote <- data[small | large, , drop = FALSE]
    output <- data_smote[, j] == large_name
    data_smote <- data_smote[, -j, drop = FALSE]
    syn_data <- smotefamily::SMOTE(data_smote, output)$syn_data

    if (nrow(syn_data) > 0) {
      syn_data$class <- NULL
      syn_data[obj$attribute] <- data[small, j][1]
      result <- rbind(result, data[small, , drop = FALSE])
      result <- rbind(result, syn_data)
    }
  }

  rownames(result) <- NULL
  result
}

iris_imb <- datasets::iris[c(1:50, 51:71, 101:111), ]
table(iris_imb$Species)

bal <- smote_custom("Species")
iris_bal <- transform(bal, iris_imb)
table(iris_bal$Species)
head(iris_bal)
