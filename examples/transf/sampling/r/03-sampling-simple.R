source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

srswor <- sample_simple(size = 10, replace = FALSE)
srswr <- sample_simple(size = 10, replace = TRUE)

sample_wor <- transform(srswor, datasets::iris$Sepal.Length)
sample_wr <- transform(srswr, datasets::iris$Sepal.Length)

sample_wor
sample_wr
