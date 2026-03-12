# Clustering - dbscan

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

# loading dataset
data(iris)

# clustering method configuration
model <- cluster_dbscan(minPts = 3)

# model fitting and labeling
model <- fit(model, iris[,1:4])
clu <- cluster(model, iris[,1:4])
table(clu)

# evaluate model using external metric
eval <- evaluate(model, clu, iris$Species)
eval
