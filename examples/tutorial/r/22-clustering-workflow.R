source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
x <- iris[, 1:4]

model <- cluster_kmeans(k = 3)
set_example_seed()
model <- daltoolbox::fit(model, x)
clu <- daltoolbox::cluster(model, x)

table(clu)
daltoolbox::evaluate(model, clu, iris$Species)

set_example_seed()
norm <- daltoolbox::fit(minmax(), iris)
iris_norm <- daltoolbox::transform(norm, iris)
x_norm <- iris_norm[, 1:4]

model_norm <- cluster_kmeans(k = 3)
set_example_seed()
model_norm <- daltoolbox::fit(model_norm, x_norm)
clu_norm <- daltoolbox::cluster(model_norm, x_norm)

table(clu_norm)
daltoolbox::evaluate(model_norm, clu_norm, iris$Species)
