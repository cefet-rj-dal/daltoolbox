source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation
# install.packages("daltoolbox")

library(daltoolbox)

iris_imb <- datasets::iris[c(1:50, 51:71, 101:111), ]
table(iris_imb$Species)

set_example_seed()
bal_random <- bal_oversampling("Species", method = "random")
iris_random <- transform(bal_random, iris_imb)
table(iris_random$Species)

set_example_seed()
bal_smote <- bal_oversampling("Species", method = "smote", k = 3)
iris_smote <- transform(bal_smote, iris_imb)
table(iris_smote$Species)
head(iris_smote)
