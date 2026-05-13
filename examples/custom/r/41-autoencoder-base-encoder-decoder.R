source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

x <- as.matrix(datasets::iris[, 1:4])

aed <- autoenc_base_ed(input_size = 4, encoding_size = 2)
aed <- fit(aed, x)
x_rec <- transform(aed, x)

dim(x_rec)
head(x_rec)
