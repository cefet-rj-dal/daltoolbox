source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

iris_imb <- datasets::iris[c(1:50, 51:71, 101:111), ]
table(iris_imb$Species)

bal_down <- sample_balance("Species", method = "down")
iris_down <- transform(bal_down, iris_imb)
table(iris_down$Species)

bal_up <- sample_balance("Species", method = "up")
iris_up <- transform(bal_up, iris_imb)
table(iris_up$Species)
