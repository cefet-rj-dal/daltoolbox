source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# installation
# install.packages("daltoolbox")

library(daltoolbox)

iris_imb <- datasets::iris[c(1:50, 51:71, 101:111), ]
table(iris_imb$Species)

set_example_seed()
bal <- bal_subsampling("Species")
iris_bal <- transform(bal, iris_imb)
table(iris_bal$Species)
head(iris_bal)
