# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
x <- iris[, 1:4]

model <- cluster_kmeans(k = 3)
model <- fit(model, x)
clu <- cluster(model, x)

table(clu)
evaluate(model, clu, iris$Species)

norm <- fit(minmax(), iris)
iris_norm <- transform(norm, iris)
x_norm <- iris_norm[, 1:4]

model_norm <- cluster_kmeans(k = 3)
model_norm <- fit(model_norm, x_norm)
clu_norm <- cluster(model_norm, x_norm)

table(clu_norm)
evaluate(model_norm, clu_norm, iris$Species)
