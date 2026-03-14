# installation
# install.packages("daltoolbox")

library(daltoolbox)

iris_imb <- datasets::iris[c(1:50, 51:71, 101:111), ]
table(iris_imb$Species)

bal <- bal_subsampling("Species", seed = 123)
iris_bal <- transform(bal, iris_imb)
table(iris_bal$Species)
head(iris_bal)
