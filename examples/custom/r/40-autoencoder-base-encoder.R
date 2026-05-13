source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

x <- as.matrix(datasets::iris[, 1:4])

enc <- autoenc_base_e(input_size = 4, encoding_size = 2)
enc <- fit(enc, x)
z <- transform(enc, x)

dim(z)
head(z)
