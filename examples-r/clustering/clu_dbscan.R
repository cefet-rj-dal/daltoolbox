# Clustering - dbscan

# installation 
#install.packages("daltoolbox")

# loading DAL
library(daltoolbox) 

# carregando conjunto de dados
data(iris)

# configuração do método de clusterização
model <- cluster_dbscan(minPts = 3)

# ajuste do modelo e rotulagem
model <- fit(model, iris[,1:4])
clu <- cluster(model, iris[,1:4])
table(clu)

# evaluate model using external metric
eval <- evaluate(model, clu, iris$Species)
eval
