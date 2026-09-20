source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/examples/seed.R"))
# install.packages("daltoolbox")

library(daltoolbox)

iris <- datasets::iris
x <- iris[, 1:4]
ref <- iris$Species
head(x)

model <- cluster_hclust(k = 3, method = "ward.D2")

model <- daltoolbox::fit(model, x)
clu <- daltoolbox::cluster(model, x)
table(clu)

eval <- daltoolbox::evaluate(model, clu, ref)
eval

grf <- daltoolbox::plot_dendrogram(model$hc, title = "Hierarchical clustering of iris")
plot(grf)
