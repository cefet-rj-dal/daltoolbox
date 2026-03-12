# installation
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris

if (requireNamespace("glmnet", quietly = TRUE)) {
  fs <- feature_selection_lasso("Sepal.Length")
  fs <- fit(fs, iris)

  print(fs$selected)
  print(fs$ranking)

  iris_fs <- transform(fs, iris)
  head(iris_fs)
}
