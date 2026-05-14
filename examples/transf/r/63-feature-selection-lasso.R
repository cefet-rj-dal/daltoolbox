source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation
# install.packages("daltoolbox")

library(daltoolbox)

data(Boston)
head(Boston)

if (requireNamespace("glmnet", quietly = TRUE)) {
  fs <- feature_selection_lasso("medv")
  set_example_seed()
  fs <- fit(fs, Boston)

  print(fs$selected)
  print(fs$ranking)

  boston_fs <- transform(fs, Boston)
  head(boston_fs)
}
