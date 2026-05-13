source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris

if (requireNamespace("leaps", quietly = TRUE)) {
  fs <- feature_selection_fss("Sepal.Length")
set_example_seed()
  fs <- fit(fs, iris)

  print(fs$selected)
  print(fs$ranking)

  iris_fs <- transform(fs, iris)
  head(iris_fs)
}
