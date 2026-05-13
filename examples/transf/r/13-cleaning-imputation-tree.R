source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

iris_na <- datasets::iris
iris_na$Sepal.Length[c(2, 10, 25)] <- NA
iris_na$Petal.Width[c(5, 11)] <- NA
iris_na$Species[c(3, 15)] <- NA

summary(iris_na)

imp <- imputation_tree(maxit = 3)
imp <- fit(imp, iris_na)
iris_imp <- transform(imp, iris_na)

summary(iris_imp$Sepal.Length)
table(iris_imp$Species, useNA = "ifany")
