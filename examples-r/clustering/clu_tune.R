# Clustering - Tune Kmeans

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox)  

data(iris)

# model training with hyperparameter search
model <- clu_tune(cluster_kmeans(k = 0),  ranges = list(k = 1:10))
model <- fit(model, iris[,1:4])
model$k

# run with best parameter
clu <- cluster(model, iris[,1:4])
table(clu)

# external evaluation using ground truth labels
eval <- evaluate(model, clu, iris$Species)
eval
