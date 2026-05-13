source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
iris$IsVersicolor <- factor(ifelse(iris$Species == "versicolor", "yes", "no"))

fs <- feature_selection_stepwise(
  "IsVersicolor",
  direction = "forward",
  family = stats::binomial
)
set_example_seed()
fs <- fit(fs, iris)

print(fs$selected)
print(fs$ranking)

iris_fs <- transform(fs, iris)
head(iris_fs)
