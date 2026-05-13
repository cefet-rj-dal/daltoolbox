source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
summary(iris)

norm <- minmax()
set_example_seed()
norm <- fit(norm, iris)
iris_norm <- transform(norm, iris)

summary(iris_norm)

fs <- feature_selection_corr(cutoff = 0.9)
set_example_seed()
fs <- fit(fs, iris_norm)

iris_fs <- transform(fs, iris_norm)

names(iris_norm)
names(iris_fs)
head(iris_fs)
