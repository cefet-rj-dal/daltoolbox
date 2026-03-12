# installation
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris

if (requireNamespace("leaps", quietly = TRUE)) {
  fs <- feature_selection_fss("Sepal.Length")
  fs <- fit(fs, iris)

  print(fs$selected)
  print(fs$ranking)

  iris_fs <- transform(fs, iris)
  head(iris_fs)
}
