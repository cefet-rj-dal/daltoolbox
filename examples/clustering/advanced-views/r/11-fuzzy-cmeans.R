source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
x <- iris[, 1:4]
ref <- iris$Species
head(x)

model <- cluster_cmeans(centers = 3, m = 2)

set_example_seed()
model <- daltoolbox::fit(model, x)
clu <- daltoolbox::cluster(model, x)
table(clu)

eval <- daltoolbox::evaluate(model, clu, ref)
eval

head(attr(clu, "membership"))
