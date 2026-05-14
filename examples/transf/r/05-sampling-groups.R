source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

sc <- sample_groups("Species", n_groups = 2)
set_example_seed()
iris_sc <- transform(sc, datasets::iris)
table(iris_sc$Species)
