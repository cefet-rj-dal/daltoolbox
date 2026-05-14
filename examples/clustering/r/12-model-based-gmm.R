source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages(c("daltoolbox", "mclust"))

library(daltoolbox)
library(mclust)

iris <- datasets::iris
x <- iris[, 1:4]
ref <- iris$Species
head(x)

model <- cluster_gmm(G = 3)

model <- daltoolbox::fit(model, x)
clu <- daltoolbox::cluster(model, x)
table(clu)

eval <- daltoolbox::evaluate(model, clu, ref)
eval
