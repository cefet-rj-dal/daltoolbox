# installation
# install.packages("daltoolbox")

library(daltoolbox)

iris_imb <- datasets::iris[c(1:50, 51:71, 101:111), ]
table(iris_imb$Species)

bal_random <- bal_oversampling("Species", method = "random", seed = 123)
iris_random <- transform(bal_random, iris_imb)
table(iris_random$Species)

bal_smote <- bal_oversampling("Species", method = "smote", k = 3, seed = 123)
iris_smote <- transform(bal_smote, iris_imb)
table(iris_smote$Species)
head(iris_smote)
