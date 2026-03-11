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
fs <- fit(fs, iris)

print(fs$selected)
print(fs$ranking)

iris_fs <- transform(fs, iris)
head(iris_fs)
