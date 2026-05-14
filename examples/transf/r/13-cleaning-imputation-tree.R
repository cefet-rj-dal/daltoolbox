source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

iris_na <- datasets::iris
iris_na$Sepal.Length[c(2, 10, 25)] <- NA

summary(iris_na)

imp <- imputation_tree("Sepal.Length")
imp <- fit(imp, iris_na)
iris_imp <- transform(imp, iris_na)

summary(iris_imp$Sepal.Length)
sum(is.na(iris_imp$Sepal.Length))
